import captum

try:
    from captum.attr import NoiseTunnel
    print('SUCCESS: captum smoothgrad was successfully imported.')
except:
    print('ERROR: captum smoothgrad was not found.')


class SmoothGradWrapper(NoiseTunnel):
    """
    SmoothGrad method using noise tunnel and saliency method
    Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). Smoothgrad: removing noise by adding noise.
    arXiv preprint arXiv:1706.03825. https://arxiv.org/abs/1706.03825
    """
    def __init__(self, model, batch_size=16, **kwargs):
        self.saliency = captum.attr.Saliency(model)
        self.batch_size = batch_size
        super().__init__(self.saliency)

    def attribute(self, inputs, target=None):
        return super().attribute(inputs,
                                 target=target,
                                 nt_samples_batch_size=self.batch_size,
                                 **{'nt_samples': 50})
