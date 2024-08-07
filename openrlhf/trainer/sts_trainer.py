import math
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openrlhf.models import SoftmaxLoss, MultipleNegativesRankingLoss, CoSENTLoss


class STSTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        self.beta = beta
        # self.loss_fn_softmax = SoftmaxLoss(self.model.model.get_input_embeddings().embedding_dim, self.args.num_labels, device='cuda:'+str(torch.cuda.current_device()))
        self.loss_fn_multi_neg_rank = MultipleNegativesRankingLoss()
        self.loss_fn_cosent = CoSENTLoss()
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        for epoch in range(self.epochs):
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            loss_score = 0
            loss_mean_score = 0
            loss_none = 0
            loss_mean_none = 0
            # train
            with profiler:
                tmp = 0
                for sentence1_ids, sentence1_masks, sentence2_ids, sentence2_masks, labels, label_types in self.train_dataloader:
                    tmp +=1
                    if tmp==3:
                        break
                    sentence1_ids = sentence1_ids.squeeze(1).to(torch.cuda.current_device())
                    sentence1_masks = sentence1_masks.squeeze(1).to(torch.cuda.current_device())
                    sentence2_ids = sentence2_ids.squeeze(1).to(torch.cuda.current_device())
                    sentence2_masks = sentence2_masks.squeeze(1).to(torch.cuda.current_device())

                    sentence1_output = self.model.model(sentence1_ids, attention_mask=sentence1_masks, output_hidden_states=True)
                    sentence2_output = self.model.model(sentence2_ids, attention_mask=sentence2_masks, output_hidden_states=True)
                    sentence1_hidden = sentence1_output.hidden_states[-1]
                    sentence2_hidden = sentence2_output.hidden_states[-1]
                    # mean pooling
                    sentence1_embed = (sentence1_hidden * sentence1_masks.unsqueeze(-1)).sum(1) / sentence1_masks.sum(1).unsqueeze(1)
                    sentence2_embed = (sentence2_hidden * sentence2_masks.unsqueeze(-1)).sum(1) / sentence2_masks.sum(1).unsqueeze(1)
                    # normalize
                    sentence1_embed = F.normalize(sentence1_embed, p=2, dim=1)
                    sentence2_embed = F.normalize(sentence2_embed, p=2, dim=1)
                    if label_types[0] == 'score':
                        labels = torch.concat(labels).to(sentence1_embed.device)
                        loss = self.loss_fn_cosent([sentence1_embed, sentence2_embed], labels)
                    elif label_types[0] == 'None':
                        loss = self.loss_fn_multi_neg_rank([sentence1_embed, sentence2_embed])
                    
                    self.strategy.backward(loss, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                    if label_types[0] == 'score':
                        loss_mean_score = loss_mean_score * 0.9 + 0.1 * loss.item()
                        loss_score = loss.item()
                    if label_types[0] == 'None':
                        loss_mean_none = loss_mean_none * 0.9 + 0.1 * loss.item()
                        loss_none = loss.item()
                    # dpo logs
                    logs_dict = {
                        'ls_score':loss_score,
                        "ls_m_score": loss_mean_score,
                        "ls_m_none":loss_mean_none,
                        'ls_none':loss_none
                    }
                    # logs/checkpoints/evaluate
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                    step_bar.update()
                    global_step += 1
            self.strategy.print(profiler.key_averages(group_by_input_shape=True))
            epoch_bar.update()
            self.strategy.print('save ckpt on epoch end, epoch: ', epoch)
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, f'epch{epoch}', args.max_ckpt_num, args.max_ckpt_mem)
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        # logs
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0 & args.save_steps > 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            loss_sum = 0
            times = 0
            for sentence1_ids, sentence1_masks, sentence2_ids, sentence2_masks, labels in self.train_dataloader:
                sentence1_ids = sentence1_ids.squeeze(1).to(torch.cuda.current_device())
                sentence1_masks = sentence1_masks.squeeze(1).to(torch.cuda.current_device())
                sentence2_ids = sentence2_ids.squeeze(1).to(torch.cuda.current_device())
                sentence2_masks = sentence2_masks.squeeze(1).to(torch.cuda.current_device())

                sentence1_output = self.model.model(sentence1_ids, attention_mask=sentence1_masks, output_hidden_states=True)
                sentence2_output = self.model.model(sentence2_ids, attention_mask=sentence2_masks, output_hidden_states=True)
                sentence1_hidden = sentence1_output.hidden_states[-1]
                sentence2_hidden = sentence2_output.hidden_states[-1]
                # mean pooling
                sentence1_embed = (sentence1_hidden * sentence1_masks.unsqueeze(-1)).sum(1) / sentence1_masks.sum(1).unsqueeze(1)
                sentence2_embed = (sentence2_hidden * sentence2_masks.unsqueeze(-1)).sum(1) / sentence2_masks.sum(1).unsqueeze(1)
                # normalize
                sentence1_embed = F.normalize(sentence1_embed, p=2, dim=1)
                sentence2_embed = F.normalize(sentence2_embed, p=2, dim=1)

                loss = self.loss_fn([sentence1_embed, sentence2_embed], labels)
                loss_sum += loss.item()
                times += 1

                logs = {
                    "eval_loss": loss_sum / times,
                }
                logs = self.strategy.all_reduce(logs)
                step_bar.set_postfix(logs)
                step_bar.update()

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

    