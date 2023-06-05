try:
    from captum.attr import Occlusion
    print('SUCCESS: captum occlusion was successfully imported.')
except:
    print('ERROR: captum occlusion was not found.')

class OcclusionWrapper:
    def __init__(self, model, batch_size=16, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.method = Occlusion(self.model)

        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', type(self.method), end='\n\n')

    def attribute(self, x, target=None, **kwargs):
        return self.method.attribute(x, target=target)