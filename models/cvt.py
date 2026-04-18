import torch
import torch.nn as nn
from timm.models import create_model

try:
    from transformers import CvtConfig, CvtForImageClassification, CvtModel
except ImportError:
    CvtConfig = None
    CvtModel = None
    CvtForImageClassification = None


def _build_hf_cvt_config(model_name: str):
    if CvtConfig is None:
        raise RuntimeError(
            "transformers is required for CvT fallback. "
            "Install it with `pip install transformers`."
        )

    if model_name == 'cvt_13':
        # Use the canonical CvT-13 architecture config from Hugging Face Hub.
        return CvtConfig.from_pretrained('microsoft/cvt-13')
    raise RuntimeError(f"Unsupported CvT model name: {model_name}")


class CvTForMFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.filter_type = config.DATA.FILTER_TYPE
        self.in_chans = config.MODEL.CVT.IN_CHANS
        self.patch_size = config.MODEL.CVT.ENCODER_STRIDE

        decode_stage = config.MODEL.CVT.DECODE_STAGE
        if decode_stage not in (1, 2, 3):
            raise ValueError(f"MODEL.CVT.DECODE_STAGE should be 1, 2, or 3, got {decode_stage}")

        out_index = decode_stage - 1
        self.out_index = out_index
        try:
            self.encoder = create_model(
                config.MODEL.CVT.NAME,
                pretrained=False,
                in_chans=self.in_chans,
                num_classes=0,
                features_only=True,
                out_indices=(out_index,),
            )
            self.use_hf_backend = False
            self.num_features = self.encoder.feature_info.channels()[-1]
        except Exception as exc:
            hf_cfg = _build_hf_cvt_config(config.MODEL.CVT.NAME)
            hf_cfg.num_channels = self.in_chans
            self.encoder = CvtModel(hf_cfg)
            self.use_hf_backend = True
            self.num_features = hf_cfg.embed_dim[out_index]

    def forward(self, x, x_fft):
        if self.filter_type == 'mfm':
            x = x_fft

        if self.use_hf_backend:
            outputs = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
            x = outputs.hidden_states[self.out_index]
        else:
            x = self.encoder(x)
            if isinstance(x, (list, tuple)):
                x = x[-1]
        if x.dim() != 4:
            raise RuntimeError(f"Expected 4D CvT feature map, got shape {tuple(x.shape)}")
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return self.encoder.no_weight_decay()
        return set()


def build_cvt(config):
    try:
        return create_model(
            config.MODEL.CVT.NAME,
            pretrained=False,
            in_chans=config.MODEL.CVT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
        )
    except Exception:
        hf_cfg = _build_hf_cvt_config(config.MODEL.CVT.NAME)
        hf_cfg.num_channels = config.MODEL.CVT.IN_CHANS
        hf_cfg.num_labels = config.MODEL.NUM_CLASSES
        model = CvtForImageClassification(hf_cfg)

        class _HFClassifierWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x, return_dict=True).logits

        return _HFClassifierWrapper(model)
