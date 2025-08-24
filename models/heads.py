import torch.nn as nn

class SingleLayerHead(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 use_activation=False, act_fn=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.use_activation = use_activation
        if use_activation and act_fn is None:
            raise ValueError(f"use_activation={use_activation} but act_fn={act_fn}")
        self.activation = (nn.Identity() if not use_activation 
                           else act_fn)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out