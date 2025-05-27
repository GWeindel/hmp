from scipy.stats import gamma, lognorm, invgauss, weibull_min
from scipy.special import gamma as gamma_func


class GammaDistribution():
    def __init__(self, shape=2):
        self.shape = shape
        self.pdf = gamma.pdf
    def scale_to_mean(self, scale):
        return scale * self.shape
    def mean_to_scale(self, mean):
        return mean / self.shape


class LognormDistribution():
    def __init__(self, shape):
        self.shape = shape
        self.pdf = lognorm.pdf
    def scale_to_mean(self, scale):
        return np.exp(scale + self.shape**2 / 2)
    def mean_to_scale(self, mean):
        return np.exp(np.log(mean) - (self.shape**2 / 2))

class WaldDistribution():
    def __init__(self, shape):
        self.shape = shape
        self.pdf = invgauss.pdf
    def scale_to_mean(self, scale):
        return scale * self.shape
    def mean_to_scale(self, mean):
        return mean / self.shape

class WeibullDistribution():  
    def __init__(self, shape):
        self.shape = shape
        self.pdf = weibull_min.pdf
        self.gamma_func = gamma_func
    def scale_to_mean(self, scale):
        return scale * self.gamma_func(1 + 1 / self.shape)
    def mean_to_scale(self, mean):
        return mean / self.gamma_func(1 + 1 / self.shape)
        
