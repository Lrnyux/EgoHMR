from .resnet import resnet
from .fcresnet import fcresnet
from .resnet_zoom_in import resnet_zoom

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'fcresnet':
        return fcresnet(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'resnet':
        return resnet(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')

def create_backbone_zoom(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'resnet':
        return resnet_zoom(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')