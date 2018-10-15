import itertools
import math
import os
import pickle
import torch
import numpy as np

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class ImageDataset(Dataset):
    def __init__(self, caption_file, image_prefix, transform=None):
        self.coco = COCO(caption_file)
        self.caption_ids = list(self.coco.anns.keys())
        self.image_prefix = image_prefix
        self.transform = transform

    def __len__(self):
        return len(self.caption_ids)

    def __getitem__(self, index):
        caption_id = self.caption_ids[index]
        image_id = self.coco.anns[caption_id]['image_id']
        filename = self.coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(self.image_prefix, filename)

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return caption_id, image_path, image


class CaptionDataset(Dataset):
    def __init__(self, caption_file, feature_dir, dictionary):
        self.dictionary = dictionary
        with open(caption_file, 'rb') as file:
            self.captions = pickle.load(file)
        with open(os.path.join(feature_dir, 'metadata.p'), 'rb') as file:
            self.image_filenames = pickle.load(file)
            # self.feature_files = {id: filename for id, (_, filename) in pickle.load(file).items()}
            # self.feature_files = {id: os.path.join(feature_dir, filename) for id, filename in pickle.load(file).items()}

        # Ignore examples that don't have both captions and features
        self.caption_ids = list(set(self.captions.keys()) & self.image_filenames.keys())
        self.caption_lengths = np.array([len(self.captions[id]) for id in self.caption_ids])

    def __len__(self):
        return len(self.caption_ids)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption_id = self.caption_ids[index]
        caption_tokens = torch.LongTensor(self.captions[caption_id])
        image_file, feature_file = self.image_filenames[caption_id]
        # with open(self.feature_files[caption_id], 'rb') as file:
        with open(feature_file, 'rb') as file:
            image_features = torch.Tensor(pickle.load(file)).float()

        return {
            'id': caption_id,
            'image_features': image_features,
            'caption_tokens': caption_tokens,
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        def merge(values, move_eos_to_beginning=False):
            max_length = max(v.size(0) for v in values)
            result = values[0].new(len(values), max_length).fill_(self.dictionary.pad_idx)

            for i, v in enumerate(values):
                if move_eos_to_beginning:
                    if v[-1] != self.dictionary.eos_idx:
                        print(v, self.dictionary.string(v))

                    assert v[-1] == self.dictionary.eos_idx
                    result[i, 0] = self.dictionary.eos_idx
                    result[i, 1:len(v)] = v[:-1]
                else:
                    result[i, :len(v)].copy_(v)
            return result

        id = torch.LongTensor([int(s['id']) for s in samples])
        image_features = torch.stack([s['image_features'] for s in samples], dim=0)
        caption_tokens = merge([s['caption_tokens'] for s in samples])
        caption_inputs = merge([s['caption_tokens'] for s in samples], move_eos_to_beginning=True)

        # Sort by descending source length
        caption_lengths = torch.LongTensor([s['caption_tokens'].numel() for s in samples])
        caption_lengths, sort_order = caption_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        image_features = image_features.index_select(0, sort_order)
        caption_tokens = caption_tokens.index_select(0, sort_order)
        caption_inputs = caption_inputs.index_select(0, sort_order)

        return {
            'id': id,
            'image_features': image_features,
            'caption_tokens': caption_tokens,
            'caption_inputs': caption_inputs,
            'caption_lengths': caption_lengths,
            'num_tokens': sum(len(s['caption_tokens']) for s in samples),
        }


class BatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=None, batch_size=None, num_shards=1, shard_id=0, shuffle=True, seed=42):
        self.dataset, self.shuffle, self.seed = dataset, shuffle, seed
        self.batch_size = batch_size if batch_size is not None else float('Inf')
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.batches = self._batch_generator()

        self.shard_len = int(math.ceil(len(self.batches) / num_shards))
        self.itr = itertools.zip_longest(
            range(self.shard_len),
            itertools.islice(self.batches, shard_id, len(self.batches), num_shards),
            fillvalue=[])

    def __len__(self):
        return self.shard_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]

    def _batch_generator(self):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        indices = indices[np.argsort(self.dataset.caption_lengths[indices], kind='mergesort')]

        batches, batch, sample_len = [], [], 0
        for idx in indices:
            batch.append(idx)
            sample_len = max(sample_len, self.dataset.caption_lengths[idx])
            num_tokens = len(batch) * sample_len
            if len(batch) == self.batch_size or num_tokens > self.max_tokens:
                batches.append(batch)
                batch, sample_len = [], 0
        if len(batch) > 0:
            batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)
        return batches
