import torch
from torch.utils.data import Dataset
from pathlib import Path
from char_lm.bpe_tokenizer import BPETokenizer


class BPEDataset(Dataset):
    def __init__(self, text_path, block_size, vocab_size=1000,
                 tokenizer_path=None):
        self.block_size = block_size

        text = Path(text_path).read_text(encoding="utf-8")

        self.tokenizer = BPETokenizer()

        if tokenizer_path and Path(tokenizer_path).exists():
            # load pre-trained tokenizer
            self.tokenizer.load(tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}")
        else:
            # train tokenizer on the corpus
            print("Training BPE tokenizer...")
            self.tokenizer.train(text, vocab_size=vocab_size)
            if tokenizer_path:
                self.tokenizer.save(tokenizer_path)

        self.vocab_size = self.tokenizer.vocab_size

        print("Encoding corpus...")
        ids = self.tokenizer.encode(text)
        self.data = torch.tensor(ids, dtype=torch.long)

        print(f"[dataset] vocab={self.vocab_size} | "
              f"tokens={len(self.data):,} | "
              f"samples={len(self):,}")

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


if __name__ == "__main__":
    dataset = BPEDataset(
        text_path      = "data/input.txt",
        block_size     = 128,
        vocab_size     = 1000,
        tokenizer_path = "char_lm/bpe_1000.json",
    )

    x, y = dataset[0]
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"x decoded: {dataset.decode(x.tolist())}")
    print(f"y decoded: {dataset.decode(y.tolist())}")




