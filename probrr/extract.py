import numpy as np
import scipy.signal as ss
from enum import Enum


class Peak(Enum):
    MIN = -1
    NONE = 0
    MAX = 1


class Extract:
    def __init__(self, t, ppg, co2=None, fc=0.49):
        self.ppg = ppg
        self.t = t                   # time vector matching the ppg signal
        self.Fs = 1 / (t[1] - t[0])  # PPG sampling frequency (Hz)
        self.fc = fc                 # digital cutoff freq for irregular-sampling reconstruction

        self.co2 = co2               # Capnograph (if it exists), assumed sampled at Fs like the PPG

        self.riiv = None
        self.riav = None
        self.rifv = None
        self.ppg_hp = None
        self.ppgMaxTimes = None
        self.T_hr = None
        self.T_rr = None


    def get_hr(self, hys=3):
        # Perform peak finding on the raw PPG with the provided hysteresis (3 usually good for
        # Capnobase).  Then, calculate the average heart rate over the length of the PPG.
        #
        # TODO: It may be desirable to perform a HR estimate on each analysis window. Pro: more
        # accurate if HR is not relatively constant (CapnoBase 0030 varies from ~105 to ~125 bpm).
        # Con: more susceptible to PPG artifacts (less averaging). Also, when comparing performance
        # to other algorithms for spot-checking, a short-window approach should be used instead;
        # it's unfair for the algorithm to be aware of the full data. In a real-time implementation,
        # the HR will presumably be calculated with tapered averaging window. But it will still be
        # necessary to decide how often to re-design the RIIV extraction filter. (With a windowed
        # sinc filter, there is a straightforward formula for the coefs, so they can be changed as
        # often as desired, without any glitching at the filter output.)
        (_, _, maxTimes, _, _, _) = self.find_peaks_hysteretic(self.t, self.ppg, hys)
        self.T_hr = np.mean(np.diff(maxTimes))  # average period of the PPG (reciprocal of heart rate)
        return 1/self.T_hr

    
    def get_riiv(self):
        # Extract the RIIV (baseline wander) signal

        if self.riiv is None:
            # The heart rate is necessary for this.
            if self.T_hr is None:
                self.get_hr()
            
            # Low-pass filter the PPG with a sharp cutoff at ~ one-half the heart rate.  The resulting
            # baseline wander is RIIV.
            #
            # Parks-McClellan (equiripple) filter design is not strictly necessary, but a replacement
            # needs to be carefully evaluated.  The filter should be quite sharp to block the main PPG
            # signal while allowing respiratory components up to Nyquist.
            lpf = ss.remez(501, [0, 0.38/self.T_hr, 0.56/self.T_hr, 0.5*self.Fs], [1, 0], fs=self.Fs)
            self.riiv = ss.convolve(self.ppg, lpf, mode='same')
            self.ppg_hp = self.ppg - self.riiv

        return (self.riiv, self.ppg_hp)


    def get_riav(self, hys=4):
        # Extract the RIAV (PPG pulse onset amplitude) signal

        if self.riav is None:
            # We start with the "de-wandered" PPG.  If RIIV has not been extracted yet, do that first.
            if self.ppg_hp is None:
                self.get_riiv()

            # Find PPG peaks for the second time, because it's more reliable once the baseline wander is removed.
            (minTimes, minVals, self.ppgMaxTimes, maxVals, _, upDeltas) = self.find_peaks_hysteretic(self.t, self.ppg_hp, hys)
            
            # Prune so that first peak is a minimum and last is a maximum
            # (ugh, this is simplified in the newer peak finder...)
            if minTimes[0] > self.ppgMaxTimes[0]:
                self.ppgMaxTimes = self.ppgMaxTimes[1:]
                maxVals = maxVals[1:]
                upDeltas = upDeltas[1:]
            if minTimes[-1] > self.ppgMaxTimes[-1]:
                minTimes = minTimes[:-1]
                minVals = minVals[:-1]
            riavTimes = (self.ppgMaxTimes+minTimes)/2

            # Reconstruct RIAV in the Karlen style, using the differences between peaks and the preceding minima
            # (the "up deltas") and then reconstructing a regularly-sampled version.
            # TODO: are these parameters really best?  could we use a different reconstruction method entirely?
            self.riav = self.marvasti_iterative_reconstruction(riavTimes, upDeltas, Fc=self.fc/self.T_hr, t=self.t, alpha=0.7, w=5.0, iters=20)[:,-1]

        # (TODO: All but RIAV are for debugging/plotting)
        return (self.riav, minTimes, minVals, self.ppgMaxTimes, maxVals, riavTimes, upDeltas)


    def get_rifv(self):
        # Extract the RIFV (PPG pulse-rate variation) signal

        if self.rifv is None:
            # We need the de-wandered PPG peak picks first, so run the RIAV extraction if it hasn't been done already.
            if self.ppgMaxTimes is None:
                self.get_riav()

            rifvTimes = self.ppgMaxTimes[:-1] + np.diff(self.ppgMaxTimes)/2
            # TODO: are these parameters really best?  could we use a different reconstruction method entirely?
            self.rifv = self.marvasti_iterative_reconstruction(rifvTimes, 1/np.diff(self.ppgMaxTimes), Fc=self.fc/self.T_hr, t=self.t, alpha=0.7, w=5.0, iters=20)[:,-1]

        # (TODO: All but RIFV are for debugging/plotting)
        return (self.rifv, rifvTimes)


    def pick_capno(self, hys=3):
        # Pick breaths using the capnography signal (if available).
        #
        # TODO: This is only an average over the entire signal.  In a real-time implementation, we can use a tapered
        # averaging window to have a continuously varying estimate.
        (co2MinTimes, co2MinVals, _, _, _ ,_) = self.find_peaks_hysteretic(self.t, self.co2, hys)
        self.T_rr = np.mean(np.diff(co2MinTimes))

        return (self.T_rr, co2MinTimes, co2MinVals)


    # TODO: The following function is an older/simpler version of the peak finder in the Sequence class!
    # We should probably figure out how to generalize the new version and use it to replace this...?
    def find_peaks_hysteretic(self, t, x, hys, initialMinMax=(0, 0)):
        # Simple peak finder which imposes a minimum distance (hysteresis) requirement between adjacent
        # maxima and minima.
        #
        # t:    time vector matching the samples in x
        # x:    samples of the function x(t)
        # hys:  minimum required hysteresis
        # initialMinMax:  the most recent minimum and maximum value of x(t) prior to time t[0]
        #
        # Returns:  times of minima, values of minima, times of maxima, values of maxima,
        #           downward (max to min) differences, upward (min to max) differences,
        #           as numpy arrays

        minInd = []     # array indices (always integer)
        minTimes = []   # actual times
        downDeltas = []
        maxInd = []     # array indices (always integer)
        maxTimes = []   # actual times
        upDeltas = []

        dirs = np.sign(np.diff(x))
        critInds = np.nonzero(np.diff(dirs))[0] + 1

        # Initial search condition
        startFlat = 0
        lastPeak = Peak.NONE

        for k in critInds:
            if dirs[k] == 0:
                # Flat spot; proceed to next critical index
                startFlat = k
                continue

            else:
                if lastPeak != Peak.MIN:
                    if len(maxInd) > 0 and x[k] > x[maxInd[-1]]:
                        # Update running maximum
                        if len(minInd) > 0:
                            upDeltas[-1] = x[k] - x[minInd[-1]]
                        else:
                            upDeltas[-1] = x[k] - initialMinMax[0]

                        maxInd[-1] = k
                        if startFlat > 0:
                            maxTimes[-1] = (t[startFlat] + t[k])/2
                        else:
                            maxTimes[-1] = t[k]

                    elif dirs[k] == 1:
                        # Possible new minimum
                        if len(maxInd) > 0:
                            delta = x[maxInd[-1]] - x[k]
                        else:
                            delta = initialMinMax[1] - x[k]

                        if delta > hys:
                            lastPeak = Peak.MIN
                            downDeltas.append(delta)
                            minInd.append(k)
                            if startFlat > 0:
                                minTimes.append((t[startFlat] + t[k])/2)
                            else:
                                minTimes.append(t[k])

                if lastPeak != Peak.MAX:
                    if len(minInd) > 0 and x[k] < x[minInd[-1]]:
                        # Update running minimum
                        if len(maxInd) > 0:
                            downDeltas[-1] = x[maxInd[-1]] - x[k]
                        else:
                            downDeltas[-1] = initialMinMax[1] - x[k]

                        minInd[-1] = k
                        if startFlat > 0:
                            minTimes[-1] = (t[startFlat] + t[k])/2
                        else:
                            minTimes[-1] = t[k]
                            
                    elif dirs[k] == -1:
                        # Possible new maximum
                        if len(minInd) > 0:
                            delta = x[k] - x[minInd[-1]]
                        else:
                            delta = x[k] - initialMinMax[0]

                        if delta > hys:
                            lastPeak = Peak.MAX
                            upDeltas.append(delta)
                            maxInd.append(k)
                            if startFlat > 0:
                                maxTimes.append((t[startFlat] + t[k])/2)
                            else:
                                maxTimes.append(t[k])

                startFlat = 0

        return (np.array(minTimes), np.array(x[minInd]), np.array(maxTimes), np.array(x[maxInd]), np.array(downDeltas), np.array(upDeltas))


    def marvasti_iterative_reconstruction(self, tk, xk, Fc, t, alpha=1.0, w=4.0, iters=1):
        # Crude iterative reconstruction of a bandlimited signal from non-uniformly spaced time samples,
        # following the algorithm of Marvasti.  Better (faster converging -> less computationally intensive)
        # methods now exist, but their implementation is far more complicated and I have not found freely
        # available code.  --mh
        #
        # xk, tk: samples of the function x(t) at the times tk
        # Fc:     cutoff frequency to use in reconstruction
        # t:      time points for reconstructed signal
        # alpha:  iteration fudge factor (damping)
        # w:      controls width of Gaussian window applied to reconstructing sincs
        # iters:  number of iterations
        #
        # Returns x(t) as a numpy array sampled at the times in t

        # Marvasti's notation: P is a low-pass operator (sinc convolution) and S is the [non-uniform] sampling
        #
        # "_u" means uniform, i.e. evaluated at the time points t
        # "_nu" means non-uniform, i.e. evaluated at the time points tk
        PSx_u  = np.zeros(t.size)
        PSx_nu = np.zeros(tk.size)
        x_u    = np.zeros((t.size, iters))
        x_nu   = np.zeros((tk.size, iters))

        for k in range(len(tk)):
            PSx_u  = PSx_u  + xk[k] * np.exp(-(t  - tk[k])*(t  - tk[k])/w) * np.sinc(2*Fc*(t  - tk[k]))
            PSx_nu = PSx_nu + xk[k] * np.exp(-(tk - tk[k])*(tk - tk[k])/w) * np.sinc(2*Fc*(tk - tk[k]))

        x_u[:,0] = alpha*PSx_u
        x_nu[:,0] = alpha*PSx_nu
        for i in range(1,iters):
            x_u[:,i]  = x_u[:,i-1]  + alpha*PSx_u
            x_nu[:,i] = x_nu[:,i-1] + alpha*PSx_nu
            for k in range(len(tk)):
                x_u[:,i]  = x_u[:,i]  - alpha*x_nu[k,i-1] * np.exp(-(t  - tk[k])*(t  - tk[k])/w) * np.sinc(2*Fc*(t  - tk[k]))
                x_nu[:,i] = x_nu[:,i] - alpha*x_nu[k,i-1] * np.exp(-(tk - tk[k])*(tk - tk[k])/w) * np.sinc(2*Fc*(tk - tk[k]))

        return x_u




    def lognormal_hybrid(self, t, tstar, tplus, shift):
        # Compute values of a scaled lognormal PDF at the points t, based on "hybrid" parameters tstar (the
        # peak or mode of the distribution), tplus (the upper half-max point), and shift.
        sigma2 = 0.72134752 * np.square(np.log((tplus - shift) / (tstar - shift)))   # constant is 1/(2 ln 2)
        mu = sigma2 + np.log(tstar - shift)
        return self.lognormal_native(t, (mu, sigma2, shift))


    def lognormal_halfmax(self, t, tminus, tstar, tplus):
        # Compute values of a scaled lognormal PDF at the points t, based on tstar (the peak or mode of the
        # distribution) and tminus/tplus (the half-max points).
        shift = (tstar*tstar - tminus*tplus) / (2*tstar - tminus - tplus)
        sigma2 = 0.18033688 * np.square(np.log((tplus - shift) / (tminus - shift)))   # constant is 1/(8 ln 2)
        mu = sigma2 + np.log(tstar - shift)
        return self.lognormal_native(t, (mu, sigma2, shift))


    def estimate_lognormal(self, t, y, tau):
        # Estimate mu and sigma-squared (the native lognormal parameters) from the data y measured at
        # times t.  An exponential decay (exp(-t/tau)) is applied backwards from the latest time in t.

        # Going back five time constants already gets us down to ~ 0.5%, so if there is more data than
        # that, ignore anything older.
        if t[-1] - t[0] > 5*tau:
            start = np.searchsorted(t, t[-1] - 5*tau)
        else:
            start = 0

        # Weight vector
        w = np.exp((t[start:] - t[-1])/tau)
        wsum = np.sum(w)
        #N = len(w)

        # See wikipedia "statistical inference" for the lognormal distribution
        mu = np.sum(w*np.log(y[start:])) / wsum
        sigma2 = np.sum(w*np.square(np.log(y[start:]) - mu)) / wsum

        return (mu, sigma2)

