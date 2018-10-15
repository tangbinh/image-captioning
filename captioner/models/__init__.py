import importlib
import os

from .model import CaptionModel


MODEL_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


def build_model(args, dictionary):
    return ARCH_MODEL_REGISTRY[args.arch].build_model(args, dictionary)


def register_model(name):
    """Decorator to register a new model"""
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model {}'.format(name))
        if not issubclass(cls, CaptionModel):
            raise ValueError('Model {} must extend {}'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls


def register_model_architecture(model_name, arch_name):
    """Decorator to register a new model architecture."""
    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError('Cannot register model architecture for unknown model type {}'.format(model_name))
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model architecture {}'.format(arch_name))
        if not callable(fn):
            raise ValueError('Model architecture must be callable {}'.format(arch_name))
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn
    return register_model_arch_fn


# Automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('captioner.models.' + module)
