import torch
import torch.nn as nn
import torch.nn.functional as F

from captioner import utils
from captioner.models import CaptionModel
from captioner.models import register_model, register_model_architecture


@register_model('show_tell')
class ShowTellModel(CaptionModel):
    def __init__(
        self, dictionary, image_dim=512, embed_dim=300, hidden_size=512, num_layers=1,
        dropout=0.2, pretrained_embedding=None,
    ):
        super().__init__(dictionary)
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.image_proj = nn.Linear(image_dim, hidden_size)

        self.embedding = pretrained_embedding
        if self.embedding is None:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        self.layers = nn.ModuleList([
            nn.LSTMCell(input_size=hidden_size + embed_dim if layer == 0 else hidden_size, hidden_size=hidden_size)
            for layer in range(num_layers)
        ])
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

    @classmethod
    def build_model(cls, args, dictionary):
        base_architecture(args)
        return cls(
            dictionary, image_dim=args.image_dim, embed_dim=args.embed_dim,
            hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
            pretrained_embedding=utils.load_embedding(args.embed_path, dictionary) if args.embed_path else None,
        )

    def forward(self, image_features, caption_inputs, incremental_state=None):
        # image_features: B x (H x W) x C
        bsz = image_features.size(0)
        image_features = self.image_proj(image_features)

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
            rnn_hiddens = [x.data.new(bsz, self.hidden_size).zero_() for i in range(len(self.layers))]
            rnn_cells = [x.data.new(bsz, self.hidden_size).zero_() for i in range(len(self.layers))]
            input_feed = image_features.mean(dim=1)

        rnn_outputs = []

        for j in range(seqlen):
            # Concatenate token embedding with output from previous time steps
            input = torch.cat([x[j, :, :], input_feed], dim=1)

            for i, rnn in enumerate(self.layers):
                # Apply recurrent cell
                rnn_hiddens[i], rnn_cells[i] = rnn(input, (rnn_hiddens[i], rnn_cells[i]))

                # Hidden state becomes the input to the next layer
                input = F.dropout(rnn_hiddens[i], p=self.dropout, training=self.training)

            # Prepare input feed for next time step
            input_feed = F.dropout(rnn_hiddens[-1], p=self.dropout, training=self.training)
            rnn_outputs.append(input_feed)

        # Cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(self, incremental_state, 'cached_state', (rnn_hiddens, rnn_cells, input_feed))

        # Collect outputs across time steps
        x = torch.cat(rnn_outputs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # Final projection
        x = self.final_proj(x)
        return x, None


@register_model_architecture('show_tell', 'show_tell')
def base_architecture(args):
    args.embed_dim = getattr(args, 'embed_dim', 300)
    args.embed_path = getattr(args, 'embed_path', None)
    args.image_dim = getattr(args, 'image_dim', 512)
    args.hidden_size = getattr(args, 'hidden_size', 512)
    args.num_layers = getattr(args, 'num_layers', 1)
    args.dropout = getattr(args, 'dropout', 0.2)
