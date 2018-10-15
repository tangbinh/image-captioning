import os
import logging
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
import sys

from skimage import transform
from matplotlib import transforms
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits import axes_grid1

from collections import defaultdict
from torch.serialization import default_restore_location


def load_embedding(embed_path, dictionary):
    """Parse an embedding text file into an torch.nn.Embedding layer."""
    embed_dict, embed_dim = {}, None
    with open(embed_path) as file:
        for line in file:
            tokens = line.rstrip().split(" ")
            embed_dim = len(tokens[1:]) if embed_dim is None else embed_dim
            embed_dict[tokens[0]] = torch.Tensor([float(weight) for weight in tokens[1:]])

    logging.info('Loaded {} / {} word embeddings'.format(
        len(set(embed_dict.keys()) & set(dictionary.words)), len(embed_dict)))
    embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)
    for idx, word in enumerate(dictionary.words):
        if word in embed_dict:
            embedding.weight.data[idx] = embed_dict[word]
    return embedding


def save_checkpoint(args, model, optimizer, lr_scheduler, epoch, valid_loss):
    if args.no_save:
        return
    os.makedirs(args.save_dir, exist_ok=True)
    last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
    save_checkpoint.last_epoch = max(last_epoch, epoch)
    prev_best = getattr(save_checkpoint, 'best_loss', float('inf'))
    save_checkpoint.best_loss = min(prev_best, valid_loss)

    state_dict = {
        'epoch': epoch,
        'val_loss': valid_loss,
        'best_loss': save_checkpoint.best_loss,
        'last_epoch': save_checkpoint.last_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
    }

    if args.epoch_checkpoints and epoch % args.save_interval == 0:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint{}_{:.3f}.pt'.format(epoch, valid_loss)))
    if valid_loss < prev_best:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint_best.pt'))
    if last_epoch < epoch:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint_last.pt'))


def load_checkpoint(args, model, optimizer, lr_scheduler):
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        save_checkpoint.best_loss = state_dict['best_loss']
        save_checkpoint.last_epoch = state_dict['last_epoch']
        logging.info('Loaded checkpoint {}'.format(checkpoint_path))
        return state_dict


def init_logging(args):
    if hasattr(args, 'distributed_rank') and args.distributed_rank != 0:
        logging.info = lambda *args, **kwargs: None
    handlers = [logging.StreamHandler()]
    if hasattr(args, 'log_file') and args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode='w'))
    logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(vars(args)))


def move_to_cuda(sample):
    if torch.is_tensor(sample):
        return sample.cuda()
    elif isinstance(sample, list):
        return [move_to_cuda(x) for x in sample]
    elif isinstance(sample, dict):
        return {key: move_to_cuda(value) for key, value in sample.items()}
    else:
        return sample


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    # Assign a unique ID to each module instance, so that incremental state is not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]
    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def plot_caption(tokens, colors, ax):
    # Split long caption into multiple lines
    def split_long_text(tokens, max_length=42):
        output, chunk, count = [], [], 0
        for token, color in zip(tokens, colors):
            count += len(token)
            if count > max_length:
                output.append(chunk)
                chunk, count = [], 0
            chunk.append((token, color))
        output.append(chunk)
        return output

    lines = split_long_text(tokens)
    for idx, line in enumerate(lines):
        transform, canvas = ax.transData, ax.figure.canvas
        for i, (tok, color) in enumerate(line):
            tok = tok.capitalize() if idx == i == 0 else ' ' + tok if tok not in ['.', ','] else tok
            offset = (idx + 1) * (0.76 - 0.18 * len(lines))
            text = ax.text(0.05, offset, tok, color=color, transform=transform, size=9)
            text.draw(canvas.get_renderer())
            ex = text.get_window_extent()
            transform = transforms.offset_copy(text._transform, x=ex.width, units='dots')


def plot_image_caption(image, output_file, system_tokens, reference_caption=None, attention=None):
    figure, axes = plt.subplots(1, 4, figsize=(18, 4))
    word_colors = [(1, 0.4, 0.4), (0.4, 1, 0.4), (0.4, 0.7, 1)]
    colormaps = ['Reds', 'Greens', 'Blues']

    tags = nltk.pos_tag(system_tokens)
    noun_indices = [idx for idx, (_, tag) in enumerate(tags) if tag in ['NN', 'NNS']]
    noun_indices = noun_indices[:len(word_colors)]

    for idx in range(len(noun_indices) + 1):
        ax = axes[idx]
        ax.set_axis_off()
        ax.imshow(image)

        caxes = []
        for position, size in [('top', '-17%'), ('bottom', '-17%')]:
            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes(position, size=size, pad=0.0)
            cax.get_xaxis().set_visible(False)
            cax.get_yaxis().set_visible(False)
            for position in ['right', 'top', 'left', 'bottom']:
                cax.spines[position].set_visible(False)
            cax.set_facecolor((0.2, 0.2, 0.2, 0.8))
            caxes.append(cax)

        properties = dict(size=14, color='white', multialignment='center')
        if idx == 0 and reference_caption is not None:
            caxes[0].add_artist(AnchoredText('Human Caption', loc=10, frameon=False, prop=properties))
            tokens = reference_caption.split(' ')
            plot_caption(tokens, ['white'] * len(tokens), caxes[1])
        else:
            caxes[0].add_artist(AnchoredText('Machine Caption', loc=10, frameon=False, prop=properties))
            colors = ['white'] * len(system_tokens)
            colors[noun_indices[idx - 1]] = word_colors[idx - 1]
            plot_caption(system_tokens, colors, caxes[1])
            attention_map = transform.pyramid_expand(attention[:, :, noun_indices[idx - 1]], upscale=16, sigma=5, multichannel=False)
            ax.imshow(attention_map, alpha=0.6, cmap=colormaps[idx - 1])

    figure.savefig(output_file, bbox_inches='tight')
    plt.close(figure)
