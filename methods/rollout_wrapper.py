import torch

try:
    from .vit_explain.vit_rollout import VITAttentionRollout
    print('SUCCESS: vit_explain was successfully imported.')
except:
    print('ERROR: vit_explain was not found.')

class RolloutWrapper():
    def __init__(self, model, discard_ratio=0.9, head_fusion='mean', **kwargs):
        self.model = model
        self.method = VITAttentionRollout(self.model, discard_ratio=discard_ratio, head_fusion=head_fusion)

        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', type(self.method), end='\n\n')

    def attribute(self, x, target=None):
        return torch.tensor(self.method(x))