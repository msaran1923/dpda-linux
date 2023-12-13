class KernelFunctor:
    def eval(self, d):
        pass

class EpanechnikovKernel(KernelFunctor):
    def eval(self, d):
        return (3.0 / 4.0) * (1.0 - d * d)
