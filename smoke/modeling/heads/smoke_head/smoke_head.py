import torch
from torch import nn

from .smoke_predictor import make_smoke_predictor
from .loss import make_smoke_loss_evaluator
from .inference import make_smoke_post_processor


class SMOKEHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKEHead, self).__init__()

        self.cfg = cfg.clone()
        self.predictor = make_smoke_predictor(cfg, in_channels)
        self.loss_evaluator = make_smoke_loss_evaluator(cfg)
        self.post_processor = make_smoke_post_processor(cfg)

    def forward(self, features, targets=None):
        x = self.predictor(features)

        print('here is the test:')

        if self.training:
            loss_heatmap, loss_regression = self.loss_evaluator(x, targets)

            return {}, dict(hm_loss=loss_heatmap,
                            reg_loss=loss_regression, )
        if not self.training:
            print('Test if this line is running:')
            result = self.post_processor(x, targets)
            # Print the parameters
            for name, param in self.post_processor.named_parameters():
                print(f"Parameter name: {name}")
                print(param)
                print("-----------")

            return result, {}


def build_smoke_head(cfg, in_channels):
    return SMOKEHead(cfg, in_channels)
