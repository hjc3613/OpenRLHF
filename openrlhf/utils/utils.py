import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import io
import json
from datasets import Dataset, interleave_datasets, load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from openrlhf.utils import DeepspeedStrategy, NoDeepspeedStrategy
from openrlhf.utils import FSDPStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload_v2(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jdict = []
    for line in tqdm(f, desc='load dataset...'):
        jdict.append(json.loads(line))
    f.close
    return jdict

def xlsload(f):
    df = pd.read_excel(f).fillna('')
    result = [dict(row) for idx, row in df.iterrows()]
    return result

def load_file(file:str):
    if file.endswith('.jsonl') or file.endswith('.txt'):
        return jload_v2(file)
    elif file.endswith('.xlsx'):
        return xlsload(file)
    elif os.path.isdir(file):
        result = []
        for sub_file in os.listdir(file):
            result.extend(load_file(os.path.join(file, sub_file)))
        return result
    
def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    if not args.close_deepspeed_or_fsdp:
        if args.use_fsdp:
            strategy = FSDPStrategy(
                seed=getattr(args, "seed", 42),
                max_norm=getattr(args, "max_norm", 1.0),
                micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
                train_batch_size=getattr(args, "train_batch_size", 128),
                zero_stage=args.zero_stage,
                bf16=getattr(args, "bf16", True),
                args=args,
            )
        else:
            strategy = DeepspeedStrategy(
                seed=getattr(args, "seed", 42),
                max_norm=getattr(args, "max_norm", 1.0),
                micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
                train_batch_size=getattr(args, "train_batch_size", 128),
                zero_stage=args.zero_stage,
                bf16=getattr(args, "bf16", True),
                args=args,
            )
    else:
        strategy = NoDeepspeedStrategy(
            seed=getattr(args, "seed", 42),
            max_norm=getattr(args, "max_norm", 1.0),
            micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
            train_batch_size=getattr(args, "train_batch_size", 128),
            zero_stage=args.zero_stage,
            bf16=getattr(args, "bf16", True),
            args=args,
        )
    return strategy


def blending_datasets(
    datasets_name,
    probabilities,
    strategy=None,
    seed=42,
    max_count_train=5000000,
    max_count_eval=100,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split=None,
    eval_split=None,
    load_ds_method='datasets.load_dataset',
):
    datasets_name = datasets_name.split(",")
    if probabilities is not None:
        probabilities = list(map(float, probabilities.split(",")))
        assert len(probabilities) == len(datasets_name)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets_name):
        if '#' in dataset:
            dataset, subset, label_type = dataset.split('#')
        else:
            subset, label_type = None, None
        if load_ds_method == 'custom':
            data = load_file(dataset)
            data = Dataset.from_list(data)
        else:
            dataset = dataset.strip()
            dataset_subfold_list = dataset.split("@")
            strategy.print(f"dataset: {dataset}")
            # local dir with python script or common local file
            if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
                (".json", ".jsonl", ".csv", ".parquet", ".txt")
            ):
                if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                    files = dataset
                    data_type = os.path.splitext(files)[1][1:]
                else:
                    path = Path(dataset)
                    script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                    extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                    files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                    strategy.print(f"script: {script}")
                    strategy.print(f"files: {files}")
                    # For dir, follow python script or first file type
                    data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
                # reformat data type
                if data_type in ["json", "jsonl"]:
                    data_type = "json"
                elif data_type == "txt":
                    data_type = "text"
                elif data_type.endswith(".py"):
                    # load local dir with python script
                    files = None
                if data_type.endswith(".py"):
                    strategy.print(f"load {dataset} with script {data_type}")
                else:
                    strategy.print(f"load {files} from {dataset}")
                data = load_dataset(data_type, data_files=files, streaming=True)
            elif len(dataset_subfold_list) == 2:
                dataset = dataset_subfold_list[0]
                subfold = dataset_subfold_list[1]
                data = load_dataset(dataset, data_dir=subfold.strip())
            elif len(dataset_subfold_list) == 1:
                dataset = dataset_subfold_list[0]
                data = load_dataset(dataset, name=subset)
            else:
                raise Exception(f"Dataset Name {dataset}: Format error")

        if train_split and train_split in data:
            # train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
            train_data = data[train_split]
        else:
            train_data = data
        max_count_train = min(max_count_train, len(train_data))
        train_data = Dataset.from_list(list(train_data.shuffle(seed=seed).take(max_count_train)))
        def add_column(example):
            example['label_type'] = label_type
            return example
        train_data = train_data.map(add_column)
        if return_eval:
            if eval_split and eval_split in data:
                # eval_data = data[eval_split].select(range(min(max_count_eval, len(data[eval_split]))))
                eval_data = data[eval_split]
                eval_data = Dataset.from_list(list(eval_data.shuffle(seed=seed).take(max_count_eval)))
            # train will contains eval? TODO
            else:
                if max_count_eval > 0:
                    split = train_data.train_test_split(test_size=min(0.2, max_count_eval/max_count_train))
                    train_data, eval_data = split['train'], split['test']
                else:
                    eval_data = Dataset.from_list([])
            eval_data = eval_data.map(add_column)
            eval_data_list.append(eval_data)

        train_data_list.append(train_data)
    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    # train_dataset = interleave_datasets(
    #     train_data_list,
    #     probabilities=probabilities,
    #     seed=seed,
    #     stopping_strategy=stopping_strategy,
    # )
    # if return_eval:
    #     eval_dataset = interleave_datasets(
    #         eval_data_list,
    #         probabilities=None,
    #         seed=seed,
    #         stopping_strategy=stopping_strategy,
    #     )
    #     return train_dataset, eval_dataset
    # else:
    #     return train_dataset
    train_dataset = concatenate_datasets(train_data_list)
    if return_eval:
        eval_dataset = concatenate_datasets(eval_data_list)
        return train_dataset, eval_dataset
    else:
        return train_dataset

