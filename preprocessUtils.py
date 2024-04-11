import matplotlib.pyplot as plt
import numpy as np
import mne





class AutoICA:
    """
    This class of functions serves as an automated noise detection tool
    in ICA components using both ECG and EOG
    """




    @staticmethod
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




    @staticmethod
    def autoICA(data, n_components=30, max_iter=1000, plot=False):

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

        phisNoise: array
        either ECG or 
        
        n_components:int - float
        ICA n_components

        max_iter: int
        maximum number of iteration

        return
        -----------
        ica: object
        final ica model

        """

        # downsampling data ==> fewer computation
        megResFil = data.copy().pick(picks=["meg"])
        megResFil.resample(200, verbose=False, n_jobs=-1)
        megResFil.filter(1, 40, verbose=False, n_jobs=-1) 


        # Extracting EOGs and ECG channels
        phisNoise = data.copy().pick(picks=["eog", "ecg"])
        phisNoise.resample(200, verbose=False, n_jobs=-1)
        phisNoise.filter(1, 40, picks=["eog", "ecg"], 
                         verbose=False, n_jobs=-1) 
        ecg, eogV, eogH = phisNoise.get_data()



        # ICA
        ica = mne.preprocessing.ICA(n_components=n_components,
                                max_iter=max_iter,
                                method="fastica",
                                random_state=42,
                                verbose=False)
        ica.fit(megResFil, verbose=False)
        

        # calculating bad ica components using automatic method
        badComponents = []
        badComponents.append(AutoICA.findComponent(ica, megResFil, ecg))
        badComponents.append(AutoICA.findComponent(ica, megResFil, eogV))
        badComponents.append(AutoICA.findComponent(ica, megResFil, eogH))

        ica.exclude = badComponents.copy()



        # you can uncomment the below codes to plot the ica results
        # if plot == True:
        #     AutoICA.plotICA(ica, data, badComponents, "grad")
        #     AutoICA.plotICA(ica, data, badComponents, "mag")

        return ica





    @staticmethod
    def plotICA(ica, data, badComponents, pick=None):

        # ocular artifact manifestation
        plt.figure()
        eogEvoked = mne.preprocessing.create_eog_epochs(data, verbose=False).average()
        eogEvoked.apply_baseline((None, -0.2))
        eogEvoked.plot_joint()
        plt.savefig("pictures/ica/eogEvoked.png", bbox_inches="tight")

        # ica sources
        ica.plot_sources(data, show_scrollbars=False)
        plt.savefig("pictures/ica/icaSources.png", bbox_inches="tight")

        # nterpolated sensor topography
        ica.plot_components(picks=badComponents, ch_type=pick)
        plt.savefig(f"pictures/ica/icaResult_{pick}.png")

        # time series
        ica.plot_overlay(data, exclude=badComponents, picks=["mag","grad"])
        plt.savefig(f"pictures/ica/icaResult2_{pick}.png")
        
        fig, ax = plt.subplots(3, 1, figsize=(30, 10))
        sig = ica.get_sources(data.copy()).get_data()[badComponents]
        ax[0].plot(sig[0][1000: 15000], color="black")
        ax[1].plot(sig[1][1000: 15000], color="black")
        ax[2].plot(sig[2][1000: 15000], color="black")
        plt.savefig("pictures/ica/icaResult3.png", bbox_inches="tight")

        print("Explained variance by ICA: ",
            ica.get_explained_variance_ratio(data).items())










