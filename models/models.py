import timm

def get_model(name, n_output, dataset=None, checkpoint=None, pretrained=True):
    
    if name == 'vit_b16':
        model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
        return model