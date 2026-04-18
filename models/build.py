from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .cvt import build_cvt
from .mfm import build_mfm


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        elif model_type == 'cvt':
            model = build_cvt(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
