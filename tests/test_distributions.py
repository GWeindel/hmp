import pytest
import numpy as np
from scipy.stats import gamma, lognorm, invgauss, weibull_min
from scipy.special import gamma as gamma_func
from hmp.distributions import Gamma, Lognorm, Wald, Weibull

@pytest.mark.parametrize("shape, scale, expected_mean", [
    (2, 3, 6),
    (4, 2, 8),
])
def test_gamma_scale_to_mean(shape, scale, expected_mean):
    dist = Gamma(shape)
    assert dist.scale_to_mean(scale) == pytest.approx(expected_mean)

@pytest.mark.parametrize("shape, mean, expected_scale", [
    (2, 6, 3),
    (4, 8, 2),
])
def test_gamma_mean_to_scale(shape, mean, expected_scale):
    dist = Gamma(shape)
    assert dist.mean_to_scale(mean) == pytest.approx(expected_scale)

@pytest.mark.parametrize("shape, scale, expected_mean", [
    (1, np.log(3), np.exp(np.log(3) + (1**2 / 2))),
    (0.5, np.log(2), np.exp(np.log(2) + (0.5**2 / 2))),
])
def test_lognorm_scale_to_mean(shape, scale, expected_mean):
    dist = Lognorm(shape)
    assert dist.scale_to_mean(scale) == pytest.approx(expected_mean)

@pytest.mark.parametrize("shape, mean, expected_scale", [
    (1, np.exp(3), np.exp(np.log(np.exp(3)) - (1**2 / 2))),
    (0.5, np.exp(2), np.exp(np.log(np.exp(2)) - (0.5**2 / 2))),
])
def test_lognorm_mean_to_scale(shape, mean, expected_scale):
    dist = Lognorm(shape)
    assert dist.mean_to_scale(mean) == pytest.approx(expected_scale)

@pytest.mark.parametrize("shape, scale, expected_mean", [
    (2, 3, 6),
    (4, 2, 8),
])
def test_wald_scale_to_mean(shape, scale, expected_mean):
    dist = Wald(shape)
    assert dist.scale_to_mean(scale) == pytest.approx(expected_mean)

@pytest.mark.parametrize("shape, mean, expected_scale", [
    (2, 6, 3),
    (4, 8, 2),
])
def test_wald_mean_to_scale(shape, mean, expected_scale):
    dist = Wald(shape)
    assert dist.mean_to_scale(mean) == pytest.approx(expected_scale)

@pytest.mark.parametrize("shape, scale, expected_mean", [
    (2, 3, 3 * gamma_func(1 + 1/2)),
    (4, 2, 2 * gamma_func(1 + 1/4)),
])
def test_weibull_scale_to_mean(shape, scale, expected_mean):
    dist = Weibull(shape)
    assert dist.scale_to_mean(scale) == pytest.approx(expected_mean)

@pytest.mark.parametrize("shape, mean, expected_scale", [
    (2, gamma_func(1 + 1/2) * 3, 3),
    (4, gamma_func(1 + 1/4) * 2, 2),
])
def test_weibull_mean_to_scale(shape, mean, expected_scale):
    dist = Weibull(shape)
    assert dist.mean_to_scale(mean) == pytest.approx(expected_scale)
