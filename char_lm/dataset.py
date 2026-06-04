from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, text_path, block_size):
        self.block_size = block_size

        text = Path(text_path).read_text(encoding="utf-8")

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.data = torch.tensor(self.encode(text), dtype=torch.long)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
    
if __name__ == "__main__":
    dataset = CharDataset("data/input.txt", block_size=8)

    x, y = dataset[0]

    print("vocab size:", dataset.vocab_size)
    print("x:", x)
    print("y:", y)
    print("x decoded:", dataset.decode(x))
    print("y decoded:", dataset.decode(y))
    print(f"[dataset] vocab={dataset.vocab_size} | tokens={len(dataset.data):,} | samples={len(dataset):,}")



