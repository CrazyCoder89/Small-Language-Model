import re
from collections import defaultdict, Counter

def get_word_freqs(text):
    word_freqs = defaultdict(int)
    words = re.findall(r'\S+', text)
    for word in words:
        token = ' '.join(list(word)) + ' </w>'
        word_freqs[token] += 1
    return word_freqs

def get_pair_freqs(word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(pair, word_freqs):
    new_word_freqs = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word, freq in word_freqs.items():
        new_word = word.replace(bigram, replacement)
        new_word_freqs[new_word] = freq

    return new_word_freqs

def train_bpe(text, vocab_size):
    # start with all unique characters as base vocabulary
    chars = set()
    for word in re.findall(r'\S+', text):
        chars.update(list(word))
    chars.add('</w>')

    vocab     = sorted(chars)        # base vocab — all individual chars
    merges    = []                   # list of merge rules in order
    word_freqs = get_word_freqs(text)

    # how many merges needed
    num_merges = vocab_size - len(vocab)

    print(f"Base vocab size: {len(vocab)}")
    print(f"Target vocab size: {vocab_size}")
    print(f"Merges to learn: {num_merges}")

    for i in range(num_merges):
        pair_freqs = get_pair_freqs(word_freqs)

        if not pair_freqs:
            break

        # find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]

        if best_freq < 2:
            break

        # merge it
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
        vocab.append(''.join(best_pair))

        if (i + 1) % 100 == 0:
            print(f"  merge {i+1}/{num_merges} | "
                  f"'{best_pair[0]}' + '{best_pair[1]}' → "
                  f"'{''.join(best_pair)}' | freq {best_freq}")

    return vocab, merges

def encode_word(word, merges):
    symbols = list(word) + ['</w>']

    for pair in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                symbols = symbols[:i] + [''.join(pair)] + symbols[i+2:]
            else:
                i += 1
    return symbols


def encode(text, merges, token_to_id):
    ids = []
    words = re.findall(r'\S+|\s+', text)

    for word in words:
        if word.strip() == '':
            # handle whitespace
            if ' ' in token_to_id:
                ids.append(token_to_id[' '])
            continue
        tokens = encode_word(word, merges)
        for token in tokens:
            if token in token_to_id:
                ids.append(token_to_id[token])
            else:
                # unknown token — fall back to characters
                for ch in token.replace('</w>', ''):
                    if ch in token_to_id:
                        ids.append(token_to_id[ch])
    return ids

def decode(ids, id_to_token):
    tokens = [id_to_token[i] for i in ids]
    text   = ''.join(tokens)
    text   = text.replace('</w>', ' ')
    text   = text.rstrip(' ')
    return text

class BPETokenizer:
    def __init__(self):
        self.vocab      = []
        self.merges     = []
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, text, vocab_size):
        self.vocab, self.merges = train_bpe(text, vocab_size)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        print(f"Trained. Final vocab size: {len(self.vocab)}")

    def encode(self, text):
        return encode(text, self.merges, self.token_to_id)

    def decode(self, ids):
        return decode(ids, self.id_to_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        import json
        data = {
            'vocab':  self.vocab,
            'merges': [list(m) for m in self.merges]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab   = data['vocab']
        self.merges  = [tuple(m) for m in data['merges']]
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        print(f"Tokenizer loaded. Vocab size: {len(self.vocab)}")

if __name__ == "__main__":
    from pathlib import Path

    text = Path("data/input.txt").read_text(encoding="utf-8")
    print(f"Corpus size: {len(text):,} characters")

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=1000)

    # test encode/decode
    test = "First Citizen:\nBefore we proceed any further, hear me speak."
    ids  = tokenizer.encode(test)
    out  = tokenizer.decode(ids)

    print(f"\nOriginal:  {test}")
    print(f"Token IDs: {ids}")
    print(f"Decoded:   {out}")
    print(f"\nChar tokenizer tokens: {len(test)}")
    print(f"BPE tokenizer tokens:  {len(ids)}")
    print(f"Compression ratio:     {len(test)/len(ids):.2f}x")

    # save
    tokenizer.save("char_lm/bpe_1000.json")



