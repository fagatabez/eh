# data.py
import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer

# removed the file loading from here entirely
# train.py handles loading now

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128):
        self.seq_len = seq_len
        ids = tokenizer.encode(text)
        self.chunks = []
        for i in range(0, len(ids) - seq_len - 1, seq_len // 2):
            chunk = ids[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]