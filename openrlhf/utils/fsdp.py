
import os
import random
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.optim import Optimizer, AdamW
from torch.utils.data.dataloader import DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import ShardingStrategy
import functools

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy
)
from openrlhf.models import Actor
from .deepspeed_utils import (
    get_optimizer_grouped_parameters
)
from .activation_checkpointing_functions import apply_fsdp_checkpointing
from ..datasets.batch_sampler import DistributedLengthBasedBatchSampler
from openrlhf.model_importer import get_model_class

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

def fsdp_auto_wrap_policy_for_llama(args, decoder_class):
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    when parallel_granularity == weight, every weight is a fsdp unit
    otherwise, the specific module name is a fsdp unit, empirical:
    when qwen2, parallel_granularity = QWenAttention-QWenMLP-RMSNorm、QWen2SdpaAttention-QWen2FlashAttention2-QWen2Attention-QWen2MLP-QWen2RMSNorm
    when qwen1, parallel_granularity = QWenBlock
    """
    # ====   use new transformer wrapper
    if args.parallel_granularity == 'decoder_layer':
        print('wrap granularity: LlamaDecoderLayer', decoder_class)
        llama_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                decoder_class,
            },
        )
    elif args.parallel_granularity == 'weight':
        print('wrap granularity: weight')
        def lambda_policy_fn(module):
            if (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
            ):
                return True
            return False

        llama_auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    else:
        minimal_units_modules = args.parallel_granularity.split('-')
        assert len(minimal_units_modules) > 0
        print('最小并行粒度：', minimal_units_modules)
        def lambda_policy_fn(module):
            module_name = module._get_name()
            if module_name in minimal_units_modules:
                # print(module_name, 'in', minimal_units_modules)
                return True
            # print(module_name, 'not in', minimal_units_modules)
            return False
        llama_auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    return llama_auto_wrap_policy

def fsdp_auto_wrap_policy_for_lora(transformer_layer_name):
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
            # FullyShardedDataParallelPlugin.get_module_class_from_name(
            #     model, transformer_layer_name
            # ),
        ),
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy

class FSDPStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        zero_stage=2,
        bf16=True,
        args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.bf16 = bf16
        self.seed = seed
        self.max_norm = max_norm
        self.adam_offload = getattr(args, "adam_offload", False)
        self.zpg = getattr(args, "zpg", 1)
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", "fp32")
        # disable_trace_cache
        self.disable_trace_cache = getattr(args, "disable_trace_cache", False)

        self.is_rlhf = False
        self.time_steps = defaultdict(int)
        self.micro_steps = 0

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f'\nlocal_rank: {local_rank} , rank: {rank}, word_size: {world_size}')
        torch.cuda.set_device(local_rank)
        clear_gpu_cache()
        setup_environ_flags(rank=rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        self.world_size = world_size
        self.args.local_rank = local_rank
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        # Optimizer
        AdamOptimizer = AdamW
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        if isinstance(model, Actor):
            model = model.model
        loss.backward()
        self.micro_steps += 1

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if isinstance(model, Actor):
            model = model.model
        if (self.micro_steps + 1) % self.accumulated_gradient == 0:
            model.clip_grad_norm_(self.args.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
    ):
        # DDP only mode, replay buffers on each rank are different.
        kwargs = {}
        if sampler is not None:
            kwargs['batch_sampler'] = sampler
        else:
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                                    replay_buffer,
                                    batch_size=batch_size,
                                    rank=dist.get_rank(),
                                    num_replicas=dist.get_world_size(),
                                    shuffle=shuffle,
                                )
        kwargs['collate_fn'] = collate_fn

        return DataLoader(
                replay_buffer,
                num_workers=0,
                pin_memory=pin_memory,
                **kwargs,
            )

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._ds_init_train_model(*arg))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, optim, scheduler):
        is_actor = isinstance(model, Actor)
        _, DecoderLayer, _ = get_model_class(self.args.model_type)
        auto_wrap_for_lora = fsdp_auto_wrap_policy_for_lora(DecoderLayer)
        
        auto_wrap_for_llama = fsdp_auto_wrap_policy_for_llama(self.args, decoder_class=DecoderLayer)
        engine = FSDP(
            model.model if is_actor else model,
            auto_wrap_policy= auto_wrap_for_lora if self.args.lora_rank > 0 else auto_wrap_for_llama,
            cpu_offload=CPUOffload(offload_params=True) if self.args.fsdp_cpu_offload else None,
            mixed_precision=None,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=self.args.use_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if self.args.use_fsdp and dist.get_rank() != 0 else None,
        )
        if self.args.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model, DecoderLayer)
        print('engine: ', engine)
        if is_actor:
            model.model = engine
        else:
            model = engine
        # recreate optimizer and scheduler for fsdp model
        optim = self.create_optimizer(model, lr=self.args.learning_rate, betas=self.args.adam_betas, weight_decay=self.args.l2)
        print('fsdp model: ', model)
        return model, optim, scheduler
    def get_ds_train_config(self, is_actor):
        return None
    
    def _ds_init_eval_model(self, model):
        is_actor = isinstance(model, Actor)
        _, DecoderLayer, _ = get_model_class(self.args.model_type)
        auto_wrap_for_lora = fsdp_auto_wrap_policy_for_lora(DecoderLayer)
        
        auto_wrap_for_llama = fsdp_auto_wrap_policy_for_llama(self.args, decoder_class=DecoderLayer)
        engine = FSDP(
            model.model if is_actor else model,
            auto_wrap_policy= auto_wrap_for_lora if self.args.lora_rank > 0 else auto_wrap_for_llama,
            # cpu_offload=CPUOffload(offload_params=True) if self.args.fsdp_cpu_offload else None,
            cpu_offload=CPUOffload(offload_params=True),
            mixed_precision=None,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=self.args.use_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if self.args.use_fsdp and dist.get_rank() != 0 else None,
        )
        if self.args.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model, DecoderLayer)

        if is_actor:
            model.model = engine
        else:
            model = engine
        print('fsdp model: ', model)
        return model

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0:
            with torch.no_grad():
                # Ensure both models are wrapped with FSDP
                assert isinstance(model, FSDP) and isinstance(model_ema, FSDP), "Both models should be wrapped with FSDP"

                # Configure FSDP for full state dict
                full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)

                # Get full state dict for both models
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    model_state_dict = model.state_dict()
                
                with FSDP.state_dict_type(model_ema, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    model_ema_state_dict = model_ema.state_dict()

                # Update EMA parameters
                for param_name, param in model_state_dict.items():
                    if param.requires_grad:
                        data = param.to(device)
                        model_ema_state_dict[param_name].copy_((1 - beta) * data + beta * model_ema_state_dict[param_name])

                # Load updated state dict back to EMA model
                with FSDP.state_dict_type(model_ema, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    model_ema.load_state_dict(model_ema_state_dict)

        # Synchronize all processes
        torch.distributed.barrier()

    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        pass

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank()

    def save_ckpt(self, model, save_dir, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True):
        # print('save_dir: ', save_dir, 'tag: ', tag)
        save_dir = save_dir
        assert isinstance(model, FSDP)
        if self.is_rank_0():
            # Check and create the directory
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            # max hard drive space limit
            MAX_SIZE = max_mem * 1024 * 1024 * 1024

            while True:
                # Get all subdirectory and modification time
                subdirs = [
                    (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                    for d in os.listdir(save_dir)
                    if os.path.isdir(os.path.join(save_dir, d))
                ]
                # Sort by modification time, oldest first
                subdirs.sort(key=lambda x: x[1])
                # Calculate the total size of all sub -directory
                total_size = 0
                for subdir, _ in subdirs:
                    for dirpath, dirnames, filenames in os.walk(subdir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)

                # If the number of subdire directors is greater than equal to max_num or the total size is greater than max_mem, the oldest Checkpoint is deleted
                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir, _ = subdirs[0]  # The oldest directory
                    if os.path.exists(oldest_dir):  # Ensure that the directory exists
                        shutil.rmtree(oldest_dir)  # Delete directory
                        self.print(f"Deleted oldest ckpt {oldest_dir}")  # The standard print function is used here
                else:
                    break

        dist.barrier()
        if self.args.lora_rank > 0:
            model.save_pretrained(self.save_dir)
        else:
            distributed_writer = dist_cp.FileSystemWriter(
                os.path.join(save_dir, tag),
            )
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
                state_dict = {"model": model.state_dict()}

                dist_cp.save_state_dict(
                    state_dict=state_dict,
                    storage_writer=distributed_writer,
                    planner=DefaultSavePlanner(),
                    
                )

        dist.barrier()

    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        pass

    
def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()

def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")

