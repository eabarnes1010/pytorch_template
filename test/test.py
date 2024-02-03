from shash.shash_torch import Shash
import scipy
import scipy.integrate as integrate
import numpy as np


epsilon = 1.0e-4
dist = Shash(np.asarray([[1.1, 0.4, 3.0, 4],]))


# test SHASH code
def f(v):
    return dist.prob(v).numpy()


cdf, _ = integrate.quad(f, -100.0, 1.5)
rvs = dist.rvs(size=100_000_000).numpy()

print(cdf, dist.cdf(1.5).numpy()[0])
print(np.median(rvs), dist.median().numpy()[0])
print(np.var(rvs), dist.var().numpy()[0])
print(scipy.stats.skew(rvs), dist.skewness().numpy()[0])
print(np.mean(rvs), dist.mean().numpy()[0])

assert np.abs(cdf - dist.cdf(1.0).numpy()) < epsilon
assert np.abs(np.median(rvs) - dist.median().numpy()) < epsilon
assert np.abs(np.var(rvs) - dist.var().numpy()) < epsilon
assert np.abs(scipy.stats.skew(rvs) - dist.skewness().numpy()) < epsilon
assert np.abs(np.mean(rvs) - dist.mean().numpy()) < epsilon
