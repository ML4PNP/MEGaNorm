import mne
import fooof as f






def computePsd(segments, freqRangeLow=3, freqRangeHigh=40, fs=1000, psdMethod="welch", psd_n_overlap=1, psd_n_fft=2, n_per_seg=2):
    """
    compute power spectrum
    """

    psds, freqs = segments.compute_psd( 
                    method = psdMethod,
                    fmin = freqRangeLow,
                    fmax = freqRangeHigh,
                    n_jobs = -1,
                    average = "mean",
                    # having at least two cycles of 1Hz activity
                    n_overlap = psd_n_overlap*fs, 
                    n_fft = psd_n_fft*fs, 
                    n_per_seg = n_per_seg*fs, 
                    verbose=False).average().get_data(return_freqs=True)
    
    return psds, freqs
    


def parameterizePsd(psds, freqs, freqRangeLow=3, freqRangeHigh=40, min_peak_height=0,
                    peak_threshold=2, peak_width_limits=[1, 12.0], aperiodic_mode="fixed"):

    """
    fooofModeling fit multiple models (group fooof) and returns 
    periodic and aperiodic signals

    parameters
    --------------
    segments: ndarray
    segmented data

    spectrumMethod: str
    the method to be used to calculate power spectrum

    fs:int
    sampling rate
    
    freqRange: list[int]
    desired frequency range to be modeled by fooof model
    default: [1, 60]

    return
    --------------
    fooofModels: object
    """  

    # fitting seperate models for each channel
    fooofModels = f.FOOOFGroup(peak_width_limits=peak_width_limits, 
                                min_peak_height=min_peak_height, 
                                peak_threshold=peak_threshold, 
                                aperiodic_mode=aperiodic_mode)
    fooofModels.fit(freqs, psds, [freqRangeLow, freqRangeHigh], n_jobs=-1)

    return fooofModels, psds, freqs