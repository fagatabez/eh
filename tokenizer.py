# tokenizer.py — turns words into numbers and back again
import json
import re
from collections import Counter

class Tokenizer:
    def __init__(self):
        # special tokens — always exist
        self.special = {
            "<pad>": 0,   # padding (fill empty space)
            "<unk>": 1,   # unknown word
            "<bos>": 2,   # beginning of sentence
            "<eos>": 3,   # end of sentence
        }
        self.word2id = dict(self.special)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.special)

    def tokenize(self, text):
        """Split text into clean tokens"""
        text = text.lower().strip()
        # split on spaces and punctuation but keep punctuation as tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def build_vocab(self, texts, max_vocab=5000):
        """Learn vocabulary from a list of texts"""
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        # take the most common words up to max_vocab
        most_common = counter.most_common(max_vocab - len(self.special))
        for word, _ in most_common:
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

        self.vocab_size = len(self.word2id)
        print(f"Vocabulary built: {self.vocab_size} tokens")

    def encode(self, text):
        """Text → list of numbers"""
        tokens = self.tokenize(text)
        ids = [self.special["<bos>"]]
        for t in tokens:
            ids.append(self.word2id.get(t, self.special["<unk>"]))
        ids.append(self.special["<eos>"])
        return ids

    def decode(self, ids):
        """List of numbers → text"""
        words = []
        for i in ids:
            word = self.id2word.get(i, "<unk>")
            if word in ("<bos>", "<pad>"):
                continue
            if word == "<eos>":
                break
            words.append(word)
        # join nicely — no leading space, punctuation attaches to previous word
        text = ""
        for w in words:
            if not text:
                text = w          # first word: no space before it
            elif re.match(r"^[^\w]", w):
                text += w         # punctuation: attach directly
            else:
                text += " " + w   # normal word: space before it
        return text.strip()

    def save(self, path="tokenizer.json"):
        with open(path, "w") as f:
            json.dump({"word2id": self.word2id}, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path="tokenizer.json"):
        with open(path) as f:
            data = json.load(f)
        self.word2id = data["word2id"]
        self.id2word = {int(v): k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        print(f"Tokenizer loaded: {self.vocab_size} tokens")