import collections
import os
import torch


class Dictionary(object):
    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>'):
        self.pad_word, self.eos_word, self.unk_word = pad, eos, unk
        self.word2idx, self.words, self.counts = {}, [], []
        self.pad_idx = self.add_word(pad)
        self.eos_idx = self.add_word(eos)
        self.unk_idx = self.add_word(unk)
        self.num_special = len(self.words)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.words[idx] if idx < len(self.words) else self.unk_word

    def index(self, word):
        return self.word2idx.get(word, self.unk_idx)

    def add_word(self, word, n=1):
        if word in self.word2idx:
            idx = self.word2idx[word]
            self.counts[idx] += n
            return idx
        else:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.words.append(word)
            self.counts.append(n)
            return idx

    def binarize(self, string, tokenizer, append_eos=True, add_if_not_exist=False, consumer=None):
        tokens = tokenizer(string)
        ids = torch.IntTensor(len(tokens) + 1 if append_eos else len(tokens))
        for i, token in enumerate(tokens):
            ids[i] = self.add_word(token) if add_if_not_exist else self.index(token)
            if consumer is not None:
                consumer(token, ids[i])
        if append_eos:
            ids[len(tokens)] = self.eos_idx
        return ids

    def string(self, tensor, bpe_symbol=None):
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)
        sentence = ' '.join(self[i] for i in tensor if i != self.eos_idx)
        if bpe_symbol is not None:
            sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
        return sentence

    def finalize(self, threshold=-1, num_words=-1):
        num_words = len(self) if num_words < 0 else num_words
        words, counts = self.words[:self.num_special], self.counts[:self.num_special]
        word2idx = dict(zip(self.words[:self.num_special], range(self.num_special)))

        counter = collections.Counter(dict(zip(self.words[self.num_special:], self.counts[self.num_special:])))
        for word, count in counter.most_common(num_words - self.num_special):
            if count >= threshold:
                word2idx[word] = len(words)
                words.append(word)
                counts.append(count)
        self.word2idx, self.words, self.counts = word2idx, words, counts

    @classmethod
    def load(cls, filename):
        """Loads the dictionary from a text file"""
        with open(filename, encoding='utf-8') as f:
            dictionary = cls()
            for line in f.readlines():
                word, count = line.rstrip().rsplit(' ', 1)
                dictionary.word2idx[word] = len(dictionary.words)
                dictionary.words.append(word)
                dictionary.counts.append(int(count))
            return dictionary

    def save(self, file):
        """Stores dictionary into a text file"""
        if isinstance(file, str):
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, 'w', encoding='utf-8') as f:
                return self.save(f)
        for word, count in zip(self.words[self.num_special:], self.counts[self.num_special:]):
            print('{} {}'.format(word, count), file=file)
