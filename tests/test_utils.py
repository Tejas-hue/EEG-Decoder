import numpy as np
from train_eegnet import _per_epoch_zscore


def test_per_epoch_zscore_zero_mean_unit_var():
    x = np.random.randn(3, 2, 5).astype(np.float32)
    z = _per_epoch_zscore(x)
    mean = z.mean(axis=2)
    std = z.std(axis=2)
    assert np.allclose(mean, 0.0, atol=1e-5)
    assert np.all(std > 0.0)