# test_wandb_offline.py
import deepspeed
import os
os.environ["WANDB_MODE"]="offline"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

import numpy as np
import pandas as pd
import wandb
from transformers import Qwen2ForCausalLM

from openrlhf.utils import blending_datasets, DeepspeedStrategy
strategy = DeepspeedStrategy(
        seed=42,
        max_norm=1.,
        micro_train_batch_size=2,
        train_batch_size=8,
        zero_stage=1,
        bf16=True,
        args=None,
    )
def is_rank_0():
    return True
strategy.is_rank_0 = is_rank_0
train_data, eval_data = blending_datasets(
        '/fl-ift/med/hujunchao/datasets/OpenOrca',
        '1.0',
        strategy,
        42,
        max_count_train=100,
        max_count_eval=10,
        train_split='train',
        eval_split='test',
    )

train_data


if __name__ == "__main__":
    # print(wandb.__version__)
    # wandb.init(project="my_wandb_test_project")
    # wandb.config["run_1"] = 1
    # wandb.log({"val_1":1})
    # wandb.log({"val_1":2})
    # wandb.log({"val_1":3})
    # wandb.summary.update({"val_2":123})
    # print("Done!")
    pass