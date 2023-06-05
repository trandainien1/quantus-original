import torch

try:
    from .chefer.baselines.ViT.ViT_explanation_generator import LRP
    from .chefer.baselines.ViT.ViT_LRP import VisionTransformer, _conv_filter, _cfg
    from .chefer.baselines.ViT.helpers import load_pretrained
    from timm.models.vision_transformer import default_cfgs as vit_cfgs
    print('SUCCESS: chefer was successfully imported.')
except:
    print('ERROR: chefer was not found.')

def vit_base_patch16_224(pretrained=True, model_name="vit_base_patch16_224", pretrained_cfg='orig_in21k_ft_in1k', **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    
    cfg = _cfg(url=vit_cfgs[model_name].cfgs[pretrained_cfg].url,nmean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    model.default_cfg = cfg

    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)

    return model.cuda()

class Chefer1Wrapper():
    def __init__(self, model, **kwargs):

        self.model = vit_base_patch16_224()
        self.model.eval()
        assert isinstance(self.model, VisionTransformer), '[ASSERT] Transformer architecture not recognised.'

        self.lrp = LRP(self.model)

        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', type(self.lrp), end='\n\n')

    def attribute(self, x, target=None):
        with torch.enable_grad():
            saliency_map = self.lrp.generate_LRP(x,  method="transformer_attribution", index=target).detach()
            return saliency_map.reshape(14, 14)