import argparse
import logging
import math
import os
import torch
import torch.nn as nn

from torch.serialization import default_restore_location
from termcolor import colored
from tqdm import tqdm

from captioner import models, utils
from captioner.data.dataset import CaptionDataset, BatchSampler
from captioner.data.dictionary import Dictionary
from captioner.generator import SequenceGenerator


def get_args():
    parser = argparse.ArgumentParser('Caption Generation')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='data-bin', help='path to data directory')
    parser.add_argument('--checkpoint-path', default='checkpoints/checkpoint_best.pt', help='path to the model file')
    parser.add_argument('--max-tokens', default=16000, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=32, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--num-workers', default=8, type=int, help='number of data workers')

    # Add generation arguments
    parser.add_argument('--beam-size', default=5, type=int, help='beam size')
    parser.add_argument('--max-len', default=200, type=int, help='maximum length of generated sequence')
    parser.add_argument('--stop-early', default='True', help='stop generation immediately after finalizing hypotheses')
    parser.add_argument('--normalize_scores', default='True', help='normalize scores by the length of the output')
    parser.add_argument('--len-penalty', default=1, type=float, help='length penalty: > 1.0 favors longer sentences')
    parser.add_argument('--unk-penalty', default=0, type=float, help='unknown word penalty: >0 produces fewer unks')
    parser.add_argument('--num-hypo', default=1, type=int, help='number of hypotheses to output')

    return parser.parse_args()


def main(args):
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args = argparse.Namespace(**{**vars(args), **vars(state_dict['args'])})
    utils.init_logging(args)

    # Load dictionary
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    logging.info('Loaded a dictionary of {} words'.format(len(dictionary)))

    # Load dataset
    test_dataset = CaptionDataset(os.path.join(args.data, 'test-tokens.p'), os.path.join(args.data, 'test-features'), dictionary)

    logging.info('Created a test dataset of {} examples'.format(len(test_dataset)))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=args.num_workers, collate_fn=test_dataset.collater, pin_memory=True,
        batch_sampler=BatchSampler(test_dataset, args.max_tokens, args.batch_size, shuffle=False, seed=args.seed))

    # Build model
    model = models.build_model(args, dictionary).cuda()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {}'.format(args.checkpoint_path))

    generator = SequenceGenerator(
        model, dictionary, beam_size=args.beam_size, maxlen=args.max_len, stop_early=eval(args.stop_early),
        normalize_scores=eval(args.normalize_scores), len_penalty=args.len_penalty, unk_penalty=args.unk_penalty,
    )

    progress_bar = tqdm(test_loader, desc='| Generation', leave=False)
    for i, sample in enumerate(progress_bar):
        sample = utils.move_to_cuda(sample)
        with torch.no_grad():
            hypos = generator.generate(sample['image_features'])

        for i, (sample_id, hypos) in enumerate(zip(sample['id'].data, hypos)):
            if sample['caption_tokens'] is not None:
                target_tokens = sample['caption_tokens'].data[i, :]
                target_tokens = target_tokens[target_tokens.ne(dictionary.pad_idx)].int().cpu()
                target_str = dictionary.string(target_tokens)
                print('T-{:<6}\t{}'.format(sample_id, colored(target_str, 'green')))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.num_hypo)]):
                hypo_tokens = hypo['tokens'].int().cpu()
                hypo_str = dictionary.string(hypo_tokens)
                alignment = hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None

                print('H-{:<6}\t{}'.format(sample_id, colored(hypo_str, 'blue')))
                if hypo['positional_scores'] is not None:
                    print('P-{:<6}\t{}'.format(sample_id, ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))))
                if alignment is not None:
                    print('A-{:<6}\t{}'.format(sample_id, ' '.join(map(lambda x: str(x.item()), alignment))))


if __name__ == '__main__':
    args = get_args()
    main(args)
