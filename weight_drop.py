# https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
# https://github.com/fastai/fastai/blob/master/fastai/text/models/awd_lstm.py (b5d459f)

# https://github.com/salesforce/awd-lstm-lm/issues/86

import torch
from torch.nn import Parameter, Module
from functools import wraps

import torch.nn as nn
import torch.nn.functional as F
import warnings

class WeightDrop(Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module:nn.Module, layer_names=[('weight_hh_l0', 0.1)]):
        super().__init__()
        self.module,self.layer_names = module,layer_names
        for layer, weight_p in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer, weight_p in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer, weight_p in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()


