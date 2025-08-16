import numpy as np
import pytest

from ppg2rr import util


@pytest.mark.parametrize(
    "y, expected, win_len",
    [
        (np.arange(1, 10), np.arange(2, 9), 3),
        (np.arange(1, 10), np.arange(3, 8), 5),
        (np.array([1, 2, 3, 4, np.nan, 6, 7, 8, 9]), np.arange(2, 9), 3),
        (np.array([1, 2, 3, 4, np.inf, 6, 7, 8, 9]), np.arange(2, 9), 3),
    ],
)
def test_moving_avg(y, expected, win_len):
    smoothed_y = util.moving_avg(y, win_len)
    assert np.all(np.isclose(smoothed_y, expected))
