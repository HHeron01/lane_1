import torch
import importlib

from .lists import lists
from mmdet3d.sparsebit.quantization.tools import fx_symbolic_trace
from mmdet3d.sparsebit.quantization.converters.prune import PruneGraph


def simplify(model: torch.fx.GraphModule):
    model = fx_symbolic_trace(model)
    model = PruneGraph().apply(model)
    for task in lists:
        module = importlib.import_module(
            ".{}".format(task), package=__package__
        ).ReplacePattern
        module().apply(model)
        model = PruneGraph().apply(model)
    return model
