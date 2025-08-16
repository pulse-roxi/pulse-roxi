import ppg2rr.capnography as capno
import numpy as np


def test_rolling_window_count():
    """
    Pass:
        returns expected values for:
            1. times array length < window length

    """
    times = np.array(
        [1.776, 5.248, 6.064, 6.944, 8.064, 12.256, 15.376, 18.352, 21.952]
    )
    window_len = 30

    t_ends, num_breaths = capno.rolling_window_count(
        times=times, window_length=window_len
    )
    assert len(t_ends) == len(num_breaths)
    assert list(t_ends) == list(range(np.ceil(times[-1]).astype("int")))
    assert list(num_breaths) == [
        0.0,
        0.0,
        2.0,
        2.0,
        2.0,
        2.0,
        4.0,
        8.0,
        8.0,
        10.0,
        10.0,
        10.0,
        10.0,
        12.0,
        12.0,
        12.0,
        14.0,
        14.0,
        14.0,
        16.0,
        16.0,
        16.0,
    ]
