import torch
import random
import numpy as np
from itertools import islice
from collections import defaultdict
from copy import deepcopy

class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        if isinstance(next(iter(data_source)), (list, tuple)):
            self.lengths = [len(d[0][0]) for d in data_source] # d[0].shape: [1, seq_len]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)
        
class STSSampler(torch.utils.data.BatchSampler):
    '''
    用于STSDataset, 向量相似度训练集分两种格式，即label_type取值为2中：label_type为score，一般用cosent loss，label_type为None，即无label，一般为(anchor, positive)对
    data_source每个element中的最后一项即为label_type
    '''
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        self.label_type_to_indices = defaultdict(list)
        for idx, item in enumerate(data_source):
            self.label_type_to_indices[item[-1]].append(idx)
        self.label_types = list(self.label_type_to_indices.keys())
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        random.seed(42)
        total_batches = 0
        label_type_to_indices = deepcopy(self.label_type_to_indices)
        label_types = deepcopy(self.label_types)
        for k, v in label_type_to_indices.items():
            total_batches += len(v) // self.batch_size
        batches = []
        for _ in range(total_batches):
            label_type = random.choice(label_types)
            indices = label_type_to_indices[label_type]
            if len(indices) < self.batch_size:
                continue
            else:
                batch_indices = random.sample(indices, self.batch_size)

            batches.append(batch_indices)
            # Remove these indices from the list to avoid reuse
            label_type_to_indices[label_type] = [
                idx for idx in label_type_to_indices[label_type] if idx not in batch_indices
            ]

            # If a label_type list becomes empty, remove it from the label_types list
            if len(label_type_to_indices[label_type]) < self.batch_size:
                label_types.remove(label_type)
        print('sts dataset iter len(batches) = ', len(batches), '\n')
        for batch in batches:
            yield batch

    def __len__(self):
        length = 0
        for k, v in self.label_type_to_indices.items():
            length += len(v) // self.batch_size
        print('sts dataset len length = ', length, '\n')
        return length

class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank
        
    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)
         
    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas
    
class DistributedSTSSmpler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = STSSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank
        
    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)
         
    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas