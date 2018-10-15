import torch.nn as nn


class CaptionModel(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, dictionary):
        """Build a new model instance."""
        raise NotImplementedError

    def forward(self, image_features, caption_inputs, **kwargs):
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state. This should be called when the order of the input has changed from the previous
        time step. A typical use case is beam search, where the input order changes between time steps based on the
        selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(incremental_state, new_order)
        self.apply(apply_reorder_incremental_state)
