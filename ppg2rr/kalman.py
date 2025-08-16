import matplotlib.pyplot as plt
import numpy as np

from ppg2rr.kalmanf import kalmanf


def init_params(waveform: np.ndarray, f0: float, fs: float, show: bool = False):
    """Initialize theta, A, and variances for the kalman filter.

    Implements equations 4,5,17,18 in [1]

    Args:
      waveform: time-series waveform, x_k in the paper
      f0: estimated resp. rate
      fs: sampling frequency
      show: If true, show plots of the estiamted waveform using calculated parameters.

    Returns:
      theta_hat: maximum likelihood estimator of the phase
      A_hat: maximum likelihood estimator of the amplitude
      var: variance of the waveform; mean sq. err. between the estimated and
            observed sinusoidal model
      b_k: measurement noise

    [1] Khreis, S. et al., Breathing Rate Estimation Using Kalman Smoother
    with Electrocardiogram and Photoplethysmogram. IEEE Trans. Biomed. Eng. 67,
    893-904 (2020).
    """
    N = len(waveform)
    k = np.arange(N)
    f_bar = f0 / fs

    # maximum likelihood estimate of amplitude and theta parameters
    # assuming noise is independent and identially distributed gaussian
    # equation 4, ML phase estimator
    theta_hat = np.arctan(
        -np.sum(waveform * np.sin(2 * np.pi * f_bar * k))
        / np.sum(waveform * np.cos(2 * np.pi * f_bar * k))
    )
    # Equation 5 (Approximate), ML amplitude estimator
    A_hat = 2 / N * np.sum(waveform * np.cos(2 * np.pi * f_bar * k + theta_hat))
    # The original long form of quation 5 is:
    # A_hat = (
    #    np.sum(waveform * np.cos(2 * np.pi * f_bar * k + theta_hat)) /
    #    np.sum(np.cos(2 * np.pi * f_bar * k + theta_hat)**2)

    # sinusoidal model & variance
    z_model = A_hat * np.cos(2 * np.pi * f_bar * k + theta_hat)
    var = 1 / N * np.sum((waveform - np.mean(waveform) - z_model) ** 2)
    b_k = z_model - waveform - np.mean(waveform)  # measurement noise

    # visualize
    if show:
        plt.figure()
        plt.plot(z_model)
        plt.plot(waveform - np.mean(waveform))
        plt.legend(["model", "input"])
        plt.show()

    return theta_hat, A_hat, var, b_k


def kalman_fusion(sig: dict[dict]):
    """Signal fusion w/ kalman filter & smoothing, based on [1-3].

    State Equations:
    X_k+1 = F*X_k + W_k; W_k ~ N(0,Q)
    Z_k = H*X_k + V_k; V_k ~ N(0,R)

    Inputs:
    sig: Nested of dictionaries containing "sensor" waveforms the following keys
        .theta - estimated phase from sinusoidal modeling
        .A - estimated amplitude from sinusoidal modeling
        .f - estimated freq from sinusoidal modeling
        .var - MSE between actual sensor waveform and modeled waveform
        .fs - sampling freq
        The outer dictionary uses integer indexing.

    Outputs:
    y  = signal derived with kalman filter
    ys = signal derived with forward-backward kalman smoother


    @ Kenny F Chou, March 2020
    New Horizon//Global Health Labs

    References:
    [1] Khreis, S et al., Breathing Rate Estimation Using Kalman Smoother
        with Electrocardiogram and Photoplethysmogram. IEEE Trans. Biomed.
        Eng. 67, 893-904 (2020).
    [2] Tarvainen, M. P., et al., A. Time-varying analysis of heart rate
        variability signals with a Kalman smoother algorithm. Physiol. Meas. 27,
        225-239 (2006).
    [3] Oliver Tracey (2021). Forward Backwards Kalman Filter
        (https://www.mathworks.com/matlabcentral/fileexchange/69889-forward-
        backwards-kalman-filter), MATLAB Central File Exchange.
        Retrieved March 8, 2021.
    """
    ########%%%%%%% set up parameters %%%%%%%%%%%%%%%%
    nSig = len(sig)
    N = np.zeros(nSig, dtype=int)
    
    # Note: A more pythonic way to do this is 
    # min_signal_len = np.min([len(signal["wav"]) for signal in sig.values()])
    for i in range(nSig):
        N[i] = len(sig[i]["wav"])
    minN = min(N)

    o_k = np.zeros(nSig)
    z = np.zeros(nSig)
    y = np.zeros(minN + 1)
    V_k = np.zeros((nSig, minN))
    s = np.zeros(minN + 1, dtype=object)

    # baseline offsets
    for i in range(nSig):
        o_k[i] = np.mean(sig[i]["wav"])
        # V_k[i, :] = sig[i]['b_k'][0:minN]  # this increases the error?

    # sinusoid parameters
    # f = normalized breathing frequency = breathing rate / sampling rate
    # theta = arbitrary phase

    A = np.zeros(nSig)
    theta = np.zeros(nSig)
    fk = np.zeros(nSig)
    f = np.zeros(nSig)
    SigVar = np.zeros(nSig)
    for i in range(nSig):
        A[i] = sig[i]["A"]
        theta[i] = sig[i]["theta"]
        fk[i] = sig[i]["f"] / sig[i]["fs"]
        SigVar[i] = sig[i]["var"]

    ########%%%%%%% state equation components %%%%%%%%%%%%%

    f = np.mean(fk)
    F1 = np.array([[1, -2 * np.pi * f], [2 * np.pi * f, 1]])
    F = np.block([[F1, np.zeros((2, nSig))], [np.zeros((nSig, 2)), np.eye(nSig)]])

    # covariance of state noise
    # Q  = [q^2 0 0 0;
    #       0 q^2 0 0;
    #       0 0 (q*o_k1)^2 0;
    #       0 0 0 (q*o_k2)^2]
    q = 0.1
    Q1 = np.diag(np.power([q, q], 2))
    Q2 = np.diag(np.power((q * o_k), 2))
    Q = np.block([[Q1, np.zeros((2, nSig))], [np.zeros((nSig, 2)), Q2]])

    # Observation matrix
    # H = [A1*cos(theta1) -A1*sin(theta1) 1 0;
    #      A2*cos(theta2) -A2*sin(theta2) 0 1];
    Acos_2d = np.expand_dims(A * np.cos(theta), axis=1)
    Asin_2d = np.expand_dims(A * np.sin(theta), axis=1)
    H = np.block([Acos_2d, -Asin_2d, np.eye(nSig)])

    R = np.diag(SigVar)

    # initial state vector
    k = 0
    x0: float = np.cos(2 * np.pi * f * k)  # breathing signal to be estimated
    v0: float = np.sin(2 * np.pi * f * k)  # negative derivative of x_k
    # x = [x0, v0, o_k1, o_k2]'; 4x1 vector.
    x: np.ndarray((4, 1)) = np.expand_dims(np.block([x0, v0, o_k]), axis=1)

    ########%%%%%%% Run Kalman Filter %%%%%%%%%%%%%%%%%
    P = Q  # covariance of the state vector estimate.

    s_init_values = {
        "A": F,
        "Q": Q,
        "H": H,
        "R": R,
        "x": x,
        "P": P,
        "u": 0,
        "B": 0,
        "z": np.zeros((nSig, 1)),
        "V": np.zeros((nSig, 1)),
    }
    s = [s_init_values.copy() for _ in range(minN)]

    z = np.zeros((nSig, 1))
    y = np.empty(minN)
    y[0] = s[0]["x"][0][0]
    for k in range(1, minN - 1):
        # observation by each "signal" at step k
        for i in range(nSig):
            z[i] = sig[i]["wav"][k]
        s[k]["z"] = z  # observation at k
        # np slicing reduces array dimensions
        # must use np.newaxis to preserve dimensionality
        s[k]["V"] = V_k[:, k, np.newaxis]
        s[k + 1] = kalmanf(s[k])
        y[k] = s[k + 1]["x"][0][0]

    """
    ########%%%%%%% Kalman Smoother %%%%%%%%%%%%%%%%%
    NOTE This chunk of code doesn't appear to work because X and s["x"] are identical,
    and (X[:, k + 1, np.newaxis] - s[k + 1]["x"]) yields a 0 vector.
    in the matlab implementation, ys = y. May be worth exploring different
    implementations of kalman filtering, such as from [3].

    backward smoothing, using the Rauch–Tung–Striebel form. Details described in [2]
    Adapted from [3].

    # fmt: off
    X = np.zeros((nSig + 2, minN)) # stores the smoothed signals
    P = np.zeros((nSig + 2, nSig + 2, minN))
    # seed last values of X and P with s.X and s.P
    X[:, minN-1, np.newaxis] = s[minN-1]["x"]
    P[:, :, minN-1] = s[minN-1]["P"]
    for k in range(minN - 2, -1, -1):
        C                   = s[k]["P"] @ F.T @ np.linalg.inv(s[k + 1]["P"])
        X[:, k, np.newaxis] = s[k]["x"] + C @ (X[:, k + 1, np.newaxis] - s[k + 1]["x"])
        P[:, :, k]          = s[k]["P"] + C @ (P[:, :, k + 1] - s[k + 1]["P"]) @ C.T
    ys = X[0, :]
    # fmt: on
    """
    ys = None

    return y, ys
