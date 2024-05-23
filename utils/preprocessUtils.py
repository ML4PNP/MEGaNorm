import matplotlib.pyplot as plt
import numpy as np
import mne






def findComponent(ica, data, phisNoise):
    """
    parameters
    -----------
    ica: object
    ica model

    data: mne.raw
    meg data

    phisNoise: array
    either ECG or 

    return
    ------------
    componentIndx: int
    index of the component with the highest Pearson 
    correlation with the signal of interest.
    """
    components = ica.get_sources(data.copy()).get_data()
    corr = np.corrcoef(components, phisNoise)[:-1, -1]
    componentIndx = np.argmax(corr)

    return componentIndx






def autoICA(data, n_components=30, max_iter=1000, IcaMethod="fastica", cutoffFreq=[1,40], whichSensor="meg"):

    """
    This function serves as an automated noise detection tool
    in ICA components. Essentially, it computes the correlation
    between each component and either the ECG or EOG signals, 
    and then returns the component with the highest Pearson 
    correlation coefficient with that signal.

    parameters
    -----------
    data: mne.raw
    meg data

    
    n_components:int - float
    ICA n_components

    max_iter: int
    maximum number of iteration

    IcaMethod: str
    ICA method

    cutoffFreq: list
    cutoff frequency for filtering data before feeding it to ICA.
    Note: this function does not filter data in place

    return
    -----------
    ica: object
    final ica model

    """
    
    megResFil = data.copy().pick(picks=[whichSensor])

    # Extracting EOGs (both vertical and horizontal sensors) and ECG channels
    phisNoise = data.copy().pick(picks=["eog", "ecg"])
    ecg, eogV, eogH = phisNoise.get_data()

    # ICA
    ica = mne.preprocessing.ICA(n_components=n_components,
                            max_iter=max_iter,
                            method=IcaMethod,
                            random_state=42,
                            verbose=False)
    ica.fit(megResFil, verbose=False)
    

    # calculating bad ica components using automatic method
    badComponents = []
    badComponents.append(findComponent(ica=ica, data=megResFil, phisNoise=ecg))
    badComponents.append(findComponent(ica=ica, data=megResFil, phisNoise=eogV))
    badComponents.append(findComponent(ica=ica, data=megResFil, phisNoise=eogH))

    ica.exclude = badComponents.copy()
    # ica.apply() changes the Raw object in-place
    ica.apply(megResFil, verbose=False)

    return megResFil








def segmentEpoch(data, tmin, tmax, fs, segmentsLength, overlap):

    # We exclude 20s from both begining and end of signals 
    # since participants usually open and close their eyes
    # in this time interval
    tmax = int(np.shape(data.get_data())[1]/fs + tmax)
    data.crop(tmin=tmin, tmax=tmax)
    
    segments = mne.make_fixed_length_epochs(data,
                                            duration=segmentsLength,
                                            overlap=overlap,
                                            reject_by_annotation=True,
                                            verbose=False)

    
    # interpolate bad segments
    return segments.load_data().interpolate_bads()














