from typing import Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(
    data,
    label_key=None,
    sentence1_key="sentence1_key",
    sentence2_key="sentence2_key",
) -> str:
    label_type = data['label_type'] 
    assert label_type in ['score', 'class', 'None']
    sentence1 = data[sentence1_key]
    sentence2 = data[sentence2_key]
    label = data.get(label_key)
    if label_type == 'None':
        label = int(-1)
    elif label_type == 'class':
        label = int(label)
    else:
        label = float(label)
    return sentence1, sentence2, label, label_type


class STSDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
    ) -> None:
        super().__init__()
        self.labels = []
        self.sentence1_lst = []
        self.sentence2_lst = []
        self.label_types = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        label_key = getattr(self.strategy.args, "label_key", None)
        sentence1_key = getattr(self.strategy.args, "sentence1_key", None)
        sentence2_key = getattr(self.strategy.args, "sentence2_key", None)

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            sentence1, sentence2, label, label_type = preprocess_data(
                data, label_key, sentence1_key, sentence2_key
            )

            self.labels.append(label)
            self.sentence1_lst.append(sentence1)
            self.sentence2_lst.append(sentence2)
            self.label_types.append(label_type)

    def __len__(self):
        length = len(self.sentence1_lst)
        return length

    def __getitem__(self, idx):
        label, sentence1, sentence2, label_type = self.labels[idx], self.sentence1_lst[idx], self.sentence2_lst[idx], self.label_types[idx]
        sentence1_token = self.tokenizer(
            sentence1,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        sentence2_token = self.tokenizer(
            sentence2,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        if isinstance(label, int):
            label = torch.LongTensor([label])
        else:
            label = torch.tensor([label])

        return (
            sentence1_token["input_ids"],
            sentence1_token["attention_mask"],
            sentence2_token["input_ids"],
            sentence2_token["attention_mask"],
            label,
            label_type
        )

    def collate_fn(self, item_list):
        sentence1_ids = []
        sentence1_masks = []
        sentence2_ids = []
        sentence2_masks = []
        labels = []
        label_types = []
        for sentence1_token, sentence1_mask, sentence2_token, sentence2_mask, label, label_type in item_list:
            sentence1_ids.append(sentence1_token)
            sentence1_masks.append(sentence1_mask)
            sentence2_ids.append(sentence2_token)
            sentence2_masks.append(sentence2_mask)
            labels.append(label)
            label_types.append(label_type)

        
        padding_side = "right"
        
        sentence1_ids = zero_pad_sequences(sentence1_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        sentence1_masks = zero_pad_sequences(sentence1_masks, side=padding_side)
        sentence2_ids = zero_pad_sequences(sentence2_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        sentence2_masks = zero_pad_sequences(sentence2_masks, side=padding_side)
        return sentence1_ids, sentence1_masks, sentence2_ids, sentence2_masks, labels, label_types
