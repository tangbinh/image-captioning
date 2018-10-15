import torch
import torch.nn as nn
import torch.nn.functional as F

from captioner import utils
from captioner.models import CaptionModel
from captioner.models import register_model, register_model_architecture


class BahdanauAttention(nn.Module):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, context_dim)
        self.context_proj = nn.Linear(context_dim, context_dim)
        self.score_proj = nn.Linear(context_dim, 1, bias=False)
        self.output_proj = nn.Linear(input_dim + context_dim, input_dim)

    def forward(self, input, context, context_mask=None):
        # input:    batch_size x input_dim
        # context:  batch_size x context_length x context_dim
        # output:   batch_size x context_length
        attn_scores = self.input_proj(input).unsqueeze(dim=1) + self.context_proj(context)
        attn_scores = self.score_proj(torch.tanh(attn_scores)).squeeze(dim=-1)

        if context_mask is not None:
            attn_scores.masked_fill_(context_mask, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Weighted sum of source hiddens
        x = (attn_scores.unsqueeze(dim=2) * context).sum(dim=1)

        # Combine with input and apply final projection
        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


@register_model('show_attend_tell')
class ShowAttendTellModel(CaptionModel):
    def __init__(
        self, dictionary, image_dim=512, embed_dim=300, hidden_size=512, num_layers=1,
        dropout=0.2, pretrained_embedding=None, use_attention=True,
    ):
        super().__init__(dictionary)
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.image_proj = nn.Linear(image_dim, hidden_size)

        self.embedding = pretrained_embedding
        if self.embedding is None:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        self.hidden_proj = nn.Linear(hidden_size, hidden_size)
        self.cell_proj = nn.Linear(hidden_size, hidden_size)

        self.layers = nn.ModuleList([
            nn.LSTMCell(input_size=hidden_size + embed_dim if layer == 0 else hidden_size, hidden_size=hidden_size)
            for layer in range(num_layers)
        ])

        self.attention = BahdanauAttention(hidden_size, hidden_size) if use_attention else None
        self.final_proj = nn.Linear(hidden_size, len(dictionary))

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--embed-dim', type=int, help='embedding dimension')
        parser.add_argument('--embed-path', help='path to pre-trained decoder embedding')
        parser.add_argument('--image-dim', type=int, help='dimension of image features')
        parser.add_argument('--hidden-size', type=int, help='hidden size')
        parser.add_argument('--num-layers', type=int, help='number of RNN layers')
        parser.add_argument('--dropout', type=float, help='dropout probability')
        parser.add_argument('--use-attention', help='whether to use attention')

    @classmethod
    def build_model(cls, args, dictionary):
        base_architecture(args)
        return cls(
            dictionary, image_dim=args.image_dim, embed_dim=args.embed_dim,
            hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
            pretrained_embedding=utils.load_embedding(args.embed_path, dictionary) if args.embed_path else None,
            use_attention=eval(args.use_attention),
        )

    def forward(self, image_features, caption_inputs, incremental_state=None):
        # image_features: B x (H x W) x C
        bsz, srclen = image_features.size()[:2]
        image_features = F.relu(self.image_proj(image_features))
        image_features = F.dropout(image_features, p=self.dropout, training=self.training)

        # Embed tokens and apply dropout
        if incremental_state is not None:
            caption_inputs = caption_inputs[:, -1:]

        seqlen = caption_inputs.size(1)
        x = self.embedding(caption_inputs)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            rnn_hiddens, rnn_cells, input_feed = cached_state
        else:
            # Initialize RNN cells with those from encoder
            rnn_hiddens = [self.hidden_proj(image_features.mean(dim=1)) for i in range(len(self.layers))]
            rnn_cells = [self.cell_proj(image_features.mean(dim=1)) for i in range(len(self.layers))]
            input_feed = image_features.mean(dim=1)

        attn_scores = x.data.new(bsz, seqlen, srclen).zero_()
        rnn_outputs = []

        for j in range(seqlen):
            # Concatenate token embedding with output from previous time step
            input = torch.cat([x[j, :, :], input_feed], dim=1)

            for i, rnn in enumerate(self.layers):
                # Apply recurrent cell
                rnn_hiddens[i], rnn_cells[i] = rnn(input, (rnn_hiddens[i], rnn_cells[i]))

                # Hidden state becomes the input to the next layer
                input = F.dropout(rnn_hiddens[i], p=self.dropout, training=self.training)

            # Prepare input feed for next time step
            if self.attention is None:
                input_feed = rnn_hiddens[-1]
            else:
                input_feed, attn_scores[:, j, :] = self.attention(rnn_hiddens[-1], image_features)

            # Prepare input feed for next time step
            input_feed = F.dropout(input_feed, p=self.dropout, training=self.training)
            rnn_outputs.append(input_feed)

        # Cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(self, incremental_state, 'cached_state', (rnn_hiddens, rnn_cells, input_feed))

        # Collect outputs across time steps
        x = torch.cat(rnn_outputs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # Final projection
        x = self.final_proj(x)
        return x, attn_scores


@register_model_architecture('show_attend_tell', 'show_attend_tell')
def base_architecture(args):
    args.embed_dim = getattr(args, 'embed_dim', 300)
    args.embed_path = getattr(args, 'embed_path', None)
    args.image_dim = getattr(args, 'image_dim', 512)
    args.hidden_size = getattr(args, 'hidden_size', 512)
    args.num_layers = getattr(args, 'num_layers', 1)
    args.dropout = getattr(args, 'dropout', 0.4)
    args.use_attention = getattr(args, 'use_attention', 'True')
