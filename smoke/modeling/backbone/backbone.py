from collections import OrderedDict

from torch import nn

from smoke.modeling import registry
from . import dla, MobileNetV2


@registry.BACKBONES.register("DLA-34-DCN")
def build_dla_backbone(cfg):
    body = dla.DLA(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
    return model

@registry.BACKBONES.register("MobileNetV2")
def build_MV2_backbone(cfg):
    body = MobileNetV2.MobileNetV2()
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
