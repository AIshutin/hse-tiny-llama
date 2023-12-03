from torch.utils.data import Dataset, Subset, DataLoader
import os
import re
import json
from tqdm import tqdm
import random
import torch
from typing import List
import pickle
import multiprocessing
import gc
import pickle


class SimpleCombiner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, story):
        return [self.tokenizer.bos_id()] + self.tokenizer.encode(story['story']) + [self.tokenizer.eos_id()]


class PlaceholderCombiner:
    def __init__(self):
        pass
        
    def __call__(self, story):
        return story['story']


def process_chunk(args):
    filename, path, combiner = args
    pattern = re.compile("data[0-9]+.json")
    if not pattern.match(filename):
        return []
    print(filename)
    with open(os.path.join(path, filename)) as file:
        j = json.load(file)
        return [combiner(el) for el in j]
    

class TinyStoriesDataset(Dataset):
    def __init__(self, combiner, path='TinyStories'):
        super().__init__()
        self.stories = []

        input_data = []
        for el in sorted(os.listdir(path)):
            input_data.append((el, path, combiner))

        pool = multiprocessing.Pool(10)
        for el in pool.map(process_chunk, input_data):
            self.stories += el
    
    def __len__(self):
        return len(self.stories)
    
    def __getitem__(self, idx):
        return self.stories[idx]


class EncodedDataset(Dataset):
    def __init__(self, encoded, max_seq_length=int(1e18)):
        print(max_seq_length, 'max_seq_length')
        self.encoded = [el[:max_seq_length] for el in encoded if len(el) > 2]
    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def get_tinystories_data(combiner, val_p=0.01):
    dataset = TinyStoriesDataset(combiner)
    random.seed(0)
    idx = list(range(len(dataset)))
    val_idx = set(random.sample(idx, k=int(len(idx) * val_p)))
    train_idx = [el for el in idx if el not in val_idx]
    train_dataset = EncodedDataset(
        [dataset[idx] for idx in train_idx]
    )
    val_dataset = EncodedDataset(
        [dataset[idx] for idx in val_idx]
    )
    return train_dataset, val_dataset


def collate_fn(batch: List[List[int]]):
    lengths = [len(el) for el in batch]
    max_length = max(lengths)
    assert(max_length > 4)
    tensor_batch = torch.zeros((len(batch), max_length), dtype=torch.long)
    for i, el in enumerate(batch):
        tensor_batch[i] = torch.tensor(el + [0] * (max_length - lengths[i]))
    return {
        "length": torch.tensor(lengths, dtype=torch.long),
        "text": tensor_batch
    }


def get_loader(split, mini_batch_tokens, num_workers=0, **kwargs):
    with open(f'TinyStories/{split}_dataset.json', 'r') as file:
        dataset = EncodedDataset(json.load(file), **kwargs)
    loader = DataLoader(dataset, collate_fn=collate_fn, \
                        num_workers=num_workers, 
                        batch_sampler=LengthSampler(dataset, 
                                                mini_batch_tokens=mini_batch_tokens))
    return loader


class LengthSampler:
    def __init__(self, dataset, mini_batch_tokens=50000):
        self.lengths = []
        for i in range(len(dataset)):
            self.lengths.append([i, len(dataset[i]), len(dataset[i])])
        self.mini_batch_tokens = mini_batch_tokens
        print(f"Dataset tokens: {sum(el[1] for el in self.lengths)}")

    def __iter__(self):
        for i in range(len(self.lengths)):
            self.lengths[i][1] = self.lengths[i][2] * random.uniform(0.95, 1.05)
        self.lengths.sort(key=lambda x: x[1])
        groups = []
        current_sum = 0
        current_group = []
        for el in self.lengths:
            current_sum += el[-1]
            current_group.append(el[0])
            if current_sum >= self.mini_batch_tokens:
                groups.append(current_group)
                current_group = []
                current_sum = 0
        if len(current_group) != 0:
            groups.append(current_group)
        return iter(groups)
    
    def __len__(self):
        cnt = 0
        for el in self:
            cnt += 1
        return cnt

    

if __name__ == "__main__":
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file='spm-tokenizer/spm.model')
    train_dataset, val_dataset = get_tinystories_data(SimpleCombiner(tokenizer))

    with open('TinyStories/train_dataset.json', 'w') as file:
        json.dump(train_dataset.encoded, file, indent=4)
    with open('TinyStories/val_dataset.json', 'w') as file:
        json.dump(val_dataset.encoded, file, indent=4)