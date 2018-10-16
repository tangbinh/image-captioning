import argparse
import collections
import logging
import nltk
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm

from captioner import utils
from captioner.data.dictionary import Dictionary
from captioner.data.dataset import ImageDataset


def get_args():
    parser = argparse.ArgumentParser('Data pre-processing')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    parser.add_argument('--data', required=True, help='path to COCO datasets')
    parser.add_argument('--dest-dir', default='data-bin', help='destination dir')

    parser.add_argument('--train-image', default='images/train2014', help='relative path to train images')
    parser.add_argument('--valid-image', default='images/val2014', help='relative path to validation images')
    parser.add_argument('--test-image', default='images/val2017', help='relative path to test images')

    parser.add_argument('--train-caption', default='annotations/captions_train2014.json', help='train captions')
    parser.add_argument('--valid-caption', default='annotations/captions_val2014.json', help='validation captions')
    parser.add_argument('--test-caption', default='annotations/captions_val2017.json', help='test captions')

    parser.add_argument('--image-size', type=int, default=256, help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--batch-size', default=80, type=int, help='batch size')
    parser.add_argument('--num-workers', default=16, type=int, help='number of workers')

    parser.add_argument('--threshold', default=0, type=int, help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words', default=-1, type=int, help='number of words to retain in dictionary')

    args = parser.parse_args()
    for name in ['train_image', 'valid_image', 'test_image', 'train_caption', 'valid_caption', 'test_caption']:
        setattr(args, name, os.path.join(args.data, getattr(args, name)))
    return args


def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.dest_dir, exist_ok=True)

    # Build dictionary
    word_tokenize = nltk.tokenize.word_tokenize
    dictionary = build_dictionary([args.train_caption], word_tokenize)
    dictionary.finalize(threshold=args.threshold, num_words=args.num_words)
    dictionary.save(os.path.join(args.dest_dir, 'dict.txt'))
    logging.info('Built a source dictionary with {} words'.format(len(dictionary)))

    make_binary_dataset(args.train_caption, os.path.join(args.dest_dir, 'train-tokens.p'), dictionary, word_tokenize)
    make_binary_dataset(args.valid_caption, os.path.join(args.dest_dir, 'valid-tokens.p'), dictionary, word_tokenize)
    make_binary_dataset(args.test_caption, os.path.join(args.dest_dir, 'test-tokens.p'), dictionary, word_tokenize)

    # Load datasets
    def load_data(split_caption, split_image):
        return ImageDataset(
            caption_file=split_caption, image_prefix=split_image,
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.RandomCrop(args.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
    train_dataset = load_data(args.train_caption, args.train_image)
    valid_dataset = load_data(args.valid_caption, args.valid_image)
    test_dataset = load_data(args.test_caption, args.test_image)

    # Extract features
    vgg = models.vgg19(pretrained=True).eval().cuda()
    model = nn.Sequential(*list(vgg.features.children())[:-2])

    extract_features(args, model, train_dataset, os.path.join(args.dest_dir, 'train-features'))
    extract_features(args, model, valid_dataset, os.path.join(args.dest_dir, 'valid-features'))
    extract_features(args, model, test_dataset, os.path.join(args.dest_dir, 'test-features'))


def build_dictionary(caption_files, tokenize):
    dictionary = Dictionary()
    for filename in caption_files:
        coco = COCO(filename)
        progress_bar = tqdm(coco.anns.values(), desc='| Build Dictionary', leave=False)
        for annotation in progress_bar:
            tokens = tokenize(annotation['caption'].lower())
            for word in tokens:
                dictionary.add_word(word)
            dictionary.add_word(dictionary.eos_word)
    return dictionary


def make_binary_dataset(caption_file, output_file, dictionary, tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()
    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    output = {}
    coco = COCO(caption_file)
    progress_bar = tqdm(coco.anns.items(), desc='| Binarize Captions', leave=False)
    for idx, (caption_id, caption) in enumerate(progress_bar):
        caption = caption['caption'].lower()
        caption_tokens = dictionary.binarize(caption, tokenize, append_eos, consumer=unk_consumer)
        output[caption_id] = caption_tokens.numpy().astype(np.int32)
        nsent, ntok = nsent + 1, ntok + len(caption_tokens)

    # Use pickle as sentence lengths vary
    with open(output_file, 'wb') as file:
        pickle.dump(output, file, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
        caption_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


def extract_features(args, model, image_dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data_loader = DataLoader(image_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    progress_bar = tqdm(data_loader, desc='| Feature Extraction', leave=False)

    filenames = {}
    for caption_ids, image_paths, sample in progress_bar:
        image_features = model(utils.move_to_cuda(sample))
        image_features = image_features.view(*image_features.size()[:-2], -1)
        # B x C x (H x W) -> B x (H x W) x C
        image_features = image_features.transpose(1, 2)
        image_features = image_features.cpu().detach().numpy().astype(np.float32)

        for id, image_path, features in zip(caption_ids.cpu().numpy().astype(np.int32), image_paths, image_features):
            filename = os.path.join(output_dir, '{}.p'.format(str(id)))
            filenames[id] = (image_path, filename)
            with open(filename, 'wb') as file:
                pickle.dump(features, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, 'metadata.p'), 'wb') as file:
        pickle.dump(filenames, file)


if __name__ == '__main__':
    args = get_args()
    utils.init_logging(args)
    main(args)
