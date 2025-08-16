import math
import numpy as np
from enum import Enum


class Peak(Enum):
    MIN = -1
    NONE = 0
    MAX = 1


class Sequence:
    def __init__(self, t, x, initialMinMax=None, pseudoPeaks=True):
        # the sequence (and corresponding probability) of the most likely peak sequence
        self.leader = ((), 0.0)

        # the running set of (sequence, probability) tuples where sequence is a tuple of peak
        # indices and probability is the sequence's estimated probability
        self.seqSet = set()

        # the list of picked peaks, starting at time zero
        self.picked = []
        
        # Find all local minima/maxima, and other peak metrics, needed by the peak-sequence finder
        if initialMinMax:
            (self.peakInds, self.peakTimes, self.peakDeltas, self.peakProms, self.firstMax) = self.find_peaks(t, x, initialMinMax, pseudoPeaks)
        else:
            (self.peakInds, self.peakTimes, self.peakDeltas, self.peakProms, self.firstMax) = self.find_peaks(t, x, (x[0], x[0]), pseudoPeaks)


    def find_peaks(self, t, x, initialMinMax, pseudoPeaks=True):
        # Simple peak finder which returns all local minima and maxima, with deltas.
        #
        # t:             time vector matching the samples in x
        # x:             samples of the function x(t)
        # initialMinMax: the most recent minimum and maximum value of x(t) prior to time t[0]
        # pseudoPeaks:   if true, include "pseudo-min/maxima" at locations where the dx/dt has a
        #                positive local minimum or a negative local maximum; these are places where
        #                the signal x "almost" has a min/max or max/min sequence, but not quite
        #               
        #
        # Returns:  indices of peaks (alternating min/max), times of peaks (matching indices),
        #           deltas, normalized prominences (see below), and a flag which is true if
        #           the first peak is a maximum, false if it's a minimum

        peakInds = []     # array indices (always integer)
        peakTimes = []    # actual times
        peakDeltas = []
                
        # Find the zero-crossings of dx/dt using first-order differencing.  (The signal x should
        # be somewhat oversampled and therefore smooth.  Noise should be LPFed beforehand.)
        dx = np.diff(x)
        dirs = np.sign(dx)

        # If we want to include pseudo-peaks, we additionally need zero-crossings of d^2 x / dt^2.
        if pseudoPeaks:
            Ddirs = np.sign(np.diff(dx))
            DcritInds = np.nonzero(np.diff(Ddirs))[0] + 1

            # The gist of the following loop is to step through the zero-crossings of d^2 x / dt^2,
            # that is, the extrema of dx/dt.  All of these are ignored except for cases where dx/dt
            # has a positive-valued local minimum or a negative-valued local maximum.  We then modify
            # the "dirs" array (sign of dx/dt) to make it look like dx/dt=0 from its preceding
            # maximum/minimum up to the minimum/maximum in question, then dx/dt=+1 or -1 as needed to
            # simulate a min/max of x, then dx/dt=0 again until its following maximum/minimum.
            #
            # When the main loop (over critInds) executes, this will cause the positive-valued local
            # minima in dx/dt to generate extra maximum-minimum pairs at the half-way points of the
            # zero runs in dirs.  Similarly, the negative-valued local maxima in dx/dt will generate
            # extra minimum-maximum pairs.
            startFlat = 0
            zeroStart = 0
            lastPeak = Peak.NONE
            pseudoPeak = Peak.NONE
            for k in DcritInds:
                if Ddirs[k] == 0:
                    startFlat = k
                    continue

                else:
                    if (Ddirs[k] == 1 and lastPeak != Peak.MIN) or (Ddirs[k] == -1 and lastPeak != Peak.MAX):
                        lastPeak = Peak.MIN if Ddirs[k] == 1 else Peak.MAX
                        
                        if startFlat > 0:
                            mid = (startFlat+k)//2
                        else:
                            mid = k

                        if pseudoPeak != Peak.NONE:
                            dirs[zeroStart:mid] = 0
                            pseudoPeak = Peak.NONE

                        if (Ddirs[k] == 1 and dx[k] > 0):
                            dirs[zeroStart:mid] = 0
                            dirs[mid] = -1
                            pseudoPeak = Peak.MIN
                        elif (Ddirs[k] == -1 and dx[k] < 0):
                            dirs[zeroStart:mid] = 0
                            dirs[mid] = 1
                            pseudoPeak = Peak.MAX

                        zeroStart = mid+1

                    startFlat = 0

        # Find the zero-crossings (actually zero-touchings) of dx, after possible modification
        # by the pseudo-peak logic above.
        critInds = np.nonzero(np.diff(dirs))[0] + 1

        # Initial search condition
        startFlat = 0
        lastPeak = Peak.NONE

        # Step through critInds with a state machine which records all of the local minima/maxima of x.
        # Each "critical index" corresponds to a change in the derivative sign, or a change to/from 0.
        # Changes from -1 to +1 indicate minima and changes from +1 to -1 indicate maxima.  Changes
        # from +1 or -1 to 0 require "memorization" of the index until the subsequent change from 0 to
        # +1 or -1, at which point a minima/maxima will be logged in the middle of the zero run.
        for k in critInds:
            if dirs[k] == 0:
                # Flat spot; proceed to next critical index
                startFlat = k
                continue

            else:
                if (dirs[k] == 1 and lastPeak != Peak.MIN) or (dirs[k] == -1 and lastPeak != Peak.MAX):
                    # New minimum of dirs[k] == 1, maximum if dirs[k] == -1
                    if startFlat > 0:
                        peakTimes.append((t[startFlat] + t[k])/2)
                        mid = (startFlat+k)//2
                    else:
                        peakTimes.append(t[k])
                        mid = k

                    if len(peakInds) > 0:
                        delta = x[mid] - x[peakInds[-1]]
                    else:
                        delta = x[mid] - (initialMinMax[1] if dirs[k] == 1 else initialMinMax[0])

                    peakDeltas.append(delta)
                    peakInds.append(mid)

                    if lastPeak == Peak.NONE:
                        firstMax = 1 if dirs[k] == 1 else 0
                    lastPeak = Peak.MIN if dirs[k] == 1 else Peak.MAX
                    
                startFlat = 0

        # Calculate normalized peak prominence for later use by the peak picker.
        #
        # "Prominence" is defined as the sum of the function value differences between a local maximum
        # (minimum) peak and its neighboring local minima (maxima).  We use a Gaussian window to smooth
        # these numbers over a user-defined time window.  Then, the normalized prominence of each peak
        # is the ratio of the single-peak prominence to the smoothed value at the same point.  (Prominence
        # is considered positive for both local minima and local maxima.)
        proms = np.zeros(len(peakInds))
        inds = np.array(peakInds)
        proms[2:-1:2] = 2*x[inds[2:-1:2]] - x[inds[1:-2:2]] - x[inds[3::2]]
        proms[1:-1:2] = x[inds[0:-2:2]] + x[inds[2::2]] - 2*x[inds[1:-1:2]]
        proms[0]  = 2*(x[inds[0]] - x[inds[1]])
        proms[-1] = 2*(x[inds[-1]] - x[inds[-2]])
        # Signs need flipping if the peak sequence was min-max-min-max-... instead of max-min-max-min-...
        if firstMax == 1:
            proms = -proms
        # Also, the last peak could be like the first, or opposite, depending on whether we have an odd or
        # even number of peaks.
        if len(peakInds) % 2 == 0:
            proms[-1] = -proms[-1]

        # Gaussian smoothing
        sigma = 5  # TODO make parameter; defines the std dev of the Gaussian smoothing function, in seconds
        den = 2*sigma*sigma  # denominator in Gaussian exponential (2*variance)
        times = np.array(peakTimes)
        for k in range(len(peakInds)):
            a = k - 1
            while a >= 0 and times[k] - times[a] < 3*sigma:
                a = a - 1
            b = k + 1
            while b < len(peakInds) and times[b] - times[k] < 3*sigma:
                b = b + 1
            
            w = np.exp(-np.square(times[a+1:b] - times[k])/den)
            proms[k] = proms[k] / ( np.sum(w*proms[a+1:b]) / np.sum(w) )

        # for debugging
        return (inds, times, np.array(peakDeltas), proms, firstMax)    


    def enumerate(self, prefix, lastPick, windowEnd, maxStep, scoreThreshold):
        # Recursive function to build the list of all valid peak sequences and compute their probabilities.
        #
        # prefix:         a (sequence, log probability) tuple defining a starting list of peak indices, which this
        #                    function will continue to expand depth-wise; for the first call use ((), 0.0)
        # lastPick:       index of the last "locked-in" peak in self.peak____ arrays, or None if this is the first
        #                    data segment
        # windowEnd:      end time for the current analysis window
        # maxStep:        the maximum breath interval; maximum time between picked peaks
        # scoreThreshold: sequence score below which a sequence path will be abandoned
        #
        if prefix[0]:  # This is a self-invocation (recursion); work from the prefix
            anchorInd = prefix[0][-1]
            anchorTime = self.peakTimes[anchorInd]   # last peak in prefix
            horizon = min(anchorTime + maxStep, windowEnd)
            k = prefix[0][-1] + 2
        else:          # This is the initial invocation
            self.leader = ((), 0.0)
            if lastPick:  # work from the last peak in the previous analysis
                anchorInd = lastPick
                anchorTime = self.peakTimes[lastPick]
                horizon = self.peakTimes[lastPick] + maxStep
                ind = np.searchsorted(self.picked, lastPick) + 1
                self.history = self.picked[:ind]
                k = lastPick + 2
            else:         # work from scratch
                anchorInd = -2 + self.firstMax
                anchorTime = 0.0
                horizon = maxStep
                self.history = []
                k = anchorInd + 2

        while self.peakTimes[k] < horizon:
            #print('prefix={}  k={}'.format(prefix[0], k))
            # New inter-breath interval if this is the next peak
            dt = self.peakTimes[k] - anchorTime
            newSeq = prefix[0] + (k,)

            historyInds = self.history + list(newSeq)
            # Find lognormal distribution parameters for timing
            if len(historyInds) > 10:
                tau = 20  # TODO make parameter
                (mu, sigma2) = self.estimate_lognormal(self.peakTimes[historyInds[1:-1]], np.diff(self.peakTimes[historyInds[:-1]]), tau)
                shift = 0
                tstar = np.exp(mu-sigma2)
                tplus = tstar*np.exp(1.1774100*np.sqrt(sigma2))   # constant is sqrt(2 ln 2)
                if tplus > 3*tstar:   # TODO make parameter
                    tplus = 3*tstar
                elif tplus < 2*tstar:   # TODO make parameter
                    tplus = 2*tstar
            elif len(historyInds) > 1:  # history exists
                shift = 0
                tstar = 2.5  # TODO make parameter
                tplus = 4    # TODO make parameter
            else:         # working from scratch (first analysis window)
                # This gives us just the right half of a lognormal (peak at zero)
                shift = -5
                tstar = 0
                tplus = 6     # probably same as default tplus above (rather than a separate parameter)

            # Maintain the running sum of log scores.  prefix[1] contains the sum of log scores for the
            # sequence thus far (or zero for the empty sequence).  The exponential of the log average
            # ("score" below) is the geometric mean of the scores from each peak in the sequence, and
            # is used to rank sequences by likelihood.
            #
            # For each peak in a sequence, the score is the product of a timing score and a prominence
            # score.  The timing score uses running information on peak spacing (inter-breath interval)
            # fit to a lognormal probability distribution.  The prominence score is based on the current
            # peak's prominence (defined above), normalized by the average peak-to-peak amplitude, then
            # run through a function TBD.
            promSig = 0.33   # TODO make parameter (relative significance of peak prominence, 0 to 1)
            logSum = prefix[1] + (1-promSig)*np.log(self.lognormal_hybrid(dt, tstar, tplus, shift) + 1e-20) + promSig*np.log(self.prominence_score(k) + 1e-20)
            score = np.exp(logSum / len(newSeq))

            # If we are within maxStep seconds of the end of the analysis window, this sequence is
            # admissible as a peak sequence for the analysis.  Add it to the set of admissible sequences.
            if windowEnd - self.peakTimes[k] < maxStep:
                #print('adding (dt {:.2f} -> {:.2e}, prominence {:.2f})  {}'.format(dt, score, prominence, newSeq))
                self.seqSet.add((newSeq, logSum))
                # Keep track of the sequence with the highest score
                if score > self.leader[1]:
                    self.leader = (newSeq, score)

            # If the score for the sequence thus far is above a user-set threshold, continue to explore
            # (by adding more peaks at the end of this sequence).  This is done with a recursive call.
            if score > scoreThreshold:
                self.enumerate((newSeq, logSum), lastPick, windowEnd, maxStep, scoreThreshold)

            # Consider the next positive peak.  Increment the index by two because the self.peak____
            # arrays alternate between positive and negative peaks.
            k = k + 2

        # If this is the original (non-recursive) invocation, use the results to append the most probable
        # peak sequence onto the running list of picked peaks, then reset the leader tracker.  In general,
        # the analysis windows will overlap, so we need to "graft" the latest picks at the correct location.
        if not prefix[0]:
            self.picked = self.history + list(self.leader[0])
        

    def lognormal_native(self, t, params):
        # Compute values of a scaled lognormal PDF(*) at the points t, based on "native" parameters in
        # the 3-tuple params = (mu, sigma2, shift).
        #
        # *This "PDF" is scaled to 1.0 at the mode/maximum, so it is not a real PDF.

        #print('mu = {:.2f}   sigma2 = {:.2f}   shift = {:.2f}'.format(params[0], params[1], params[2]))

        if np.ndim(t) > 0:
            f = np.zeros(t.shape)
            f[t>params[2]] = np.exp(params[0] - 0.5*params[1]) * np.exp(-0.5*np.square(np.log(t[t>params[2]] - params[2]) - params[0])/params[1]) / (t[t>params[2]] - params[2])
        else:
            if t > params[2]:
                f = np.exp(params[0] - 0.5*params[1]) * np.exp(-0.5*np.square(np.log(t - params[2]) - params[0])/params[1]) / (t - params[2])
            else:
                f = 0.0

        return f

        
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


    def prominence_score(self, ind):
        # For peak index ind, return the prominence score used in peak-sequence scoring.
        #
        # We want a sigmoidal function which is easily parameterized for centering and steepness;
        # the error function (erf) is ideal for this.
        return 0.5 + 0.5*math.erf(0.7 * (self.peakProms[ind] - 1.0))

    
    def test(self, seq, lastPick):
        # For testing:  re-score a particular peak sequence, with debug output
        #
        # seq:            a sequence of peaks to try
        # lastPick:       index of the last "locked-in" peak in self.peak____ arrays, or None if this is the first
        #                    data segment

        if lastPick:  # work from the last peak in the previous analysis
            anchorInd = lastPick
            anchorTime = self.peakTimes[lastPick]
            ind = np.searchsorted(self.picked, lastPick) + 1
            history = self.picked[:ind]
        else:         # work from scratch
            anchorInd = -2 + self.firstMax
            anchorTime = 0.0
            history = []

        logSum = 0.0
        count = 1

        print('Analyzing sequence {}'.format(seq))

        for k in range(len(seq)):

            if k == 0:
                dt = self.peakTimes[seq[k]] - anchorTime
            else:
                dt = self.peakTimes[seq[k]] - self.peakTimes[seq[k-1]]

            history.append(seq[k])

            # Find lognormal distribution parameters for timing
            if len(history) > 10:
                tau = 20  # TODO make parameter
                (mu, sigma2) = self.estimate_lognormal(self.peakTimes[history[1:-1]], np.diff(self.peakTimes[history[:-1]]), tau)
                shift = 0
                tstar = np.exp(mu-sigma2)
                tplusOrig = tstar*np.exp(1.1774100*np.sqrt(sigma2))   # constant is sqrt(2 ln 2)
                if tplusOrig > 3*tstar:    # TODO make parameter
                    tplus = 3*tstar
                elif tplusOrig < 2*tstar:  # TODO make parameter
                    tplus = 2*tstar
                else:
                    tplus = tplusOrig
            elif len(history) > 1:  # history exists
                shift = 0
                tstar = 2.5  # TODO make parameter
                tplus = 4    # TODO make parameter
                tplusOrig = 5
            else:         # working from scratch (first analysis window)
                # This gives us just the right half of a lognormal (peak at zero)
                shift = -5
                tstar = 0
                tplus = 6      # probably same as default tplus above (rather than a separate parameter)
                tplusOrig = 6

            print('history is {}'.format(history))

            # Maintain the running sum of log scores.  prefix[1] contains the sum of log scores for the
            # sequence thus far (or zero for the empty sequence).  The exponential of the log average
            # ("score" below) is the geometric mean of the scores from each peak in the sequence, and
            # is used to rank sequences by likelihood.
            #
            # For each peak in a sequence, the score is the product of a timing score and a prominence
            # score.  The timing score uses running information on peak spacing (inter-breath interval)
            # fit to a lognormal probability distribution.  The prominence score is based on the current
            # peak's prominence (defined above), normalized by the average peak-to-peak amplitude, then
            # run through a function TBD.
            promSig = 0.33   # TODO make parameter (relative significance of peak prominence, 0 to 1)
            tmp = (1-promSig)*np.log(self.lognormal_hybrid(dt, tstar, tplus, shift) + 1e-20) + promSig*np.log(self.prominence_score(seq[k]) + 1e-20)
            logSum = logSum + tmp
            score = np.exp(logSum / count)

            print('dt {:.2f}   shift {:.2f}   tstar {:.2f}   tplus {:.2f} (orig {:.2f})   -> timescore {:.3f}'.format(dt, shift, tstar, tplus, tplusOrig, np.power(self.lognormal_hybrid(dt, tstar, tplus, shift), 1-promSig)))
            print('prom {:.2f} -> promscore {:.2f}'.format(self.peakProms[seq[k]], np.power(self.prominence_score(seq[k]), promSig)))
            print('logscore change {:.2f}   logscore sum {:.2f}   score {:.2f}\n'.format(tmp, logSum, score))
            count = count + 1        