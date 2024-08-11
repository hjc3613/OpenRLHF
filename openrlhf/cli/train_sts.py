import argparse
import math
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from torch import distributed as dist
from transformers.trainer import get_scheduler

from openrlhf.datasets import STSDataset
from openrlhf.models import Actor
from openrlhf.trainer import STSTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.datasets.batch_sampler import DistributedLengthBasedBatchSamplerMultiType

def create_dataloader(dataset, strategy, args, tokenizer):
    dataset, subset, type = dataset.split('#')
    



def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        device_map=args.device_map,
        model_type=args.model_type,
        low_cpu=args.use_fsdp
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count_train=args.max_samples_train,
        max_count_eval=args.max_samples_eval,
        stopping_strategy="all_exhausted",
        train_split=args.train_split,
        eval_split=args.eval_split,
        load_ds_method=args.load_ds_method
    )
    train_dataset = STSDataset(
        train_data, tokenizer, args.max_len, strategy
    )
    eval_dataset = STSDataset(
        eval_data, tokenizer, args.max_len, strategy
    )

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
        batch_sampler=DistributedLengthBasedBatchSamplerMultiType(
            train_dataset,
            batch_size=args.micro_train_batch_size,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
            shuffle=True,
        )
    )

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn,
        batch_sampler=DistributedLengthBasedBatchSamplerMultiType(
            train_dataset,
            batch_size=args.micro_train_batch_size,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
            shuffle=True,
        )
    )
    
    # train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    # eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    
    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
        # strategy.load_checkpoint(args.save_path + '/rm_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = STSTrainer(
        model=model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        beta=args.beta,
        max_epochs=args.max_epochs,
    )

    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # new added
    parser.add_argument("--freeze_strategy", type=str, default=None)
    parser.add_argument("--transformer_layers_path", type=str, default="model.model.layers")
    parser.add_argument("--use_fsdp", default=False, action="store_true")
    parser.add_argument("--parallel_granularity", type=str, default='QWenBlock', help='Qwen2DecoderLayer、QWenBlock ...')
    parser.add_argument("--fsdp_cpu_offload", default=False, action="store_true")
    parser.add_argument("--fsdp_activation_checkpointing", default=False, action='store_true')
    parser.add_argument("--model_type", type=str, required=True, help='qwen1, qwen2, llama, mixtral...')
    parser.add_argument("--load_ds_method", type=str, default="datasets.load_dataset",choices=["datasets.load_dataset", "custom"])
    
    parser.add_argument("--device_map", default=None, type=str, help="device_map, when close deepspeed for debug, set device_map to auto for split model on multiple gpu")
    
    parser.add_argument("--close_deepspeed_or_fsdp", action="store_true", default=False, help="close deepspeed, for single process debug mode")
    
    parser.add_argument('--num_labels', default=2, help='class numbers for sts')
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)

    # DPO
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Custom dataset
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default=None, help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--label_key", type=str, default="score")
    parser.add_argument("--sentence1_key", type=str, default="sentence1")
    parser.add_argument("--sentence2_key", type=str, default="sentence2")
    parser.add_argument("--input_template", type=str, default="{}")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--max_samples_train", type=int, default=1e8, help="Max number of samples for train")
    parser.add_argument("--max_samples_eval", type=int, default=1e8, help="Max number of samples for eval")
    parser.add_argument("--max_len", type=int, default=512)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    train(args)
