import argparse
import logging
import multiprocessing
import os
import torch
import torch.nn as nn

from tqdm import tqdm

from captioner import models, utils
from captioner.data.dataset import CaptionDataset, BatchSampler
from captioner.data.dictionary import Dictionary
from captioner.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


def get_args():
    parser = argparse.ArgumentParser('Image Captioning')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='data-bin', help='path to data directory')
    parser.add_argument('--max-tokens', default=16000, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=120, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--num-workers', default=multiprocessing.cpu_count(), type=int, help='number of data workers')

    # Add model arguments
    parser.add_argument('--arch', default='show_attend_tell', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=100, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=5, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.99, type=float, help='momentum factor')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--lr-shrink', default=0.2, type=float, help='learning rate shrink factor for annealing')
    parser.add_argument('--min-lr', default=1e-6, type=float, help='minimum learning rate')

    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):
    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported.')
    torch.manual_seed(args.seed)
    utils.init_logging(args)

    # Load dictionary
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    logging.info('Loaded a dictionary of {} words'.format(len(dictionary)))

    # Load datasets
    train_dataset = CaptionDataset(os.path.join(args.data, 'train-tokens.p'), os.path.join(args.data, 'train-features'), dictionary)
    logging.info('Created a train dataset of {} examples'.format(len(train_dataset)))
    valid_dataset = CaptionDataset(os.path.join(args.data, 'valid-tokens.p'), os.path.join(args.data, 'valid-features'), dictionary)
    logging.info('Created a validation dataset of {} examples'.format(len(valid_dataset)))

    # Build model and criterion
    model = models.build_model(args, dictionary).cuda()
    logging.info('Built a model with {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    criterion = nn.CrossEntropyLoss(ignore_index=dictionary.pad_idx, reduction='sum').cuda()

    # Build an optimizer and a learning rate schedule
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, min_lr=args.min_lr, factor=args.lr_shrink)

    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer, lr_scheduler)
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1
    optimizer.param_groups[0]['lr'] = args.lr

    for epoch in range(last_epoch + 1, args.max_epoch):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=args.num_workers, collate_fn=train_dataset.collater, pin_memory=True,
            batch_sampler=BatchSampler(train_dataset, args.max_tokens, args.batch_size, shuffle=True, seed=args.seed))

        model.train()
        stats = {'loss': 0., 'lr': 0., 'num_tokens': 0., 'batch_size': 0., 'grad_norm': 0., 'clip': 0.}
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

        for i, sample in enumerate(progress_bar):
            # Forward and backward pass
            sample = utils.move_to_cuda(sample)
            output, _ = model(sample['image_features'], sample['caption_inputs'])

            loss = criterion(output.view(-1, output.size(-1)), sample['caption_tokens'].view(-1))
            optimizer.zero_grad()
            loss.backward()

            # Normalize gradients by number of tokens and perform clipping
            for name, param in model.named_parameters():
                param.grad.data.div_(sample['num_tokens'])
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            # Update statistics for progress bar
            stats['loss'] += loss.item() / sample['num_tokens']
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += sample['num_tokens'] / len(sample['caption_inputs'])
            stats['batch_size'] += len(sample['caption_inputs'])
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()}, refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))

        # Adjust learning rate based on validation loss
        valid_loss = validate(args, model, criterion, valid_dataset, epoch)
        lr_scheduler.step(valid_loss)

        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, lr_scheduler, epoch, valid_loss)
        if optimizer.param_groups[0]['lr'] <= args.min_lr:
            logging.info('Done training!')
            break


def validate(args, model, criterion, valid_dataset, epoch):
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, num_workers=args.num_workers, collate_fn=valid_dataset.collater, pin_memory=True,
        batch_sampler=BatchSampler(valid_dataset, args.max_tokens, args.batch_size, shuffle=False, seed=args.seed))

    model.eval()
    stats = {'valid_loss': 0, 'num_tokens': 0, 'batch_size': 0}
    progress_bar = tqdm(valid_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

    for i, sample in enumerate(progress_bar):
        sample = utils.move_to_cuda(sample)
        output, _ = model(sample['image_features'], sample['caption_inputs'])
        with torch.no_grad():
            loss = criterion(output.view(-1, output.size(-1)), sample['caption_tokens'].view(-1))

        stats['valid_loss'] += loss.item() / sample['num_tokens']
        stats['num_tokens'] += sample['num_tokens'] / len(sample['caption_inputs'])
        stats['batch_size'] += len(sample['caption_inputs'])
        progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()}, refresh=True)

    logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
        value / len(progress_bar)) for key, value in stats.items())))
    return stats['valid_loss'] / len(progress_bar)


if __name__ == '__main__':
    args = get_args()
    main(args)
