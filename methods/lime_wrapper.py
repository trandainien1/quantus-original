try:
    from captum.attr import Lime
    print('SUCCESS: captum lime was successfully imported.')
except:
    print('ERROR: captum lime was not found.')

class LimeWrapper:
    def __init__(self, model, batch_size=16, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.method = Lime(self.model)

        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', type(self.method), end='\n\n')

    def attribute(self, x, target=None, **kwargs):
        return self.method.attribute(x, target=target)