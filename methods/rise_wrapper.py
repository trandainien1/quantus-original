import torch

try:
    from .rise.explanations import RISE
    print('SUCCESS: rise was successfully imported.')
except:
    print('ERROR: rise was not found.')

class RISEWrapper():
    def __init__(self, model, n_masks=4000, input_size=224, batch_size=2, **kwargs):
        self.model = model
        self.rise = RISE(self.model, (input_size, input_size), gpu_batch=batch_size)

        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', type(self.rise), end='\n\n')

        self.rise.generate_masks(N=n_masks, s=8, p1=0.1)
        self.input_size = input_size

    def attribute(self, x, target=None):
        with torch.no_grad():
            return self.rise(x)[target].view(self.input_size, self.input_size).detach()