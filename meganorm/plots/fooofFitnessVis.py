import matplotlib.pyplot as plt
import fooof as f
import pickle
import glob
import numpy as np
import os
import math


def averaged_psds_component(paths):
    """
    Reads fooof results (in pickle) and averages PSDs, aperiodic, and periodic components.
    Returns the averaged components along with frequencies.
    """
    # Initialize accumulators as arrays
    psds_sum, aperiodics_sum, periodics_sum = (np.zeros(75), np.zeros(75), np.zeros(75))
    total_channels = 0

    for path in paths:
        with open(path, "rb") as file:
            fooofres = pickle.load(file)

        # Save frequencies only once
        if total_channels == 0:
            freqs = fooofres[2].copy()

        # Check all channels for valid aperiodic components
        for i in range(fooofres[1].shape[0]):
            fm = fooofres[0].get_fooof(ind=i)

            # Skip if aperiodic model not available
            if math.isnan(fm._ap_fit[0]):
                continue

            aperiodic = 10**fm._ap_fit
            aperiodics_sum += aperiodic
            periodics_sum += fooofres[1][i, :] - aperiodic
            psds_sum += fooofres[1][i, :]
            total_channels += 1

    # Compute averages
    psds_avg = psds_sum / total_channels
    aperiodics_avg = aperiodics_sum / total_channels
    periodics_avg = periodics_sum / total_channels

    return freqs, psds_avg, periodics_avg, aperiodics_avg


def plot_knee_fixed_compare_psd(
    freqs, psds, fixed_aperiodics, knee_aperiodics, save_path
):
    """
    Plot the aperiodic, periodic and original PSDs
    (not in log log space)
    """
    freqs, psds, _, knee_aperiodics = averaged_psds_component(knee_path)
    _, _, _, fixed_aperiodics = averaged_psds_component(fixed_path)

    plt.figure(figsize=(8, 7))
    plt.plot(freqs, psds, color="black", linewidth=2, label="Original PSD")
    plt.plot(freqs, fixed_aperiodics, color="#A9561E", label="Aperiodic - Fixed mode")
    plt.plot(freqs, knee_aperiodics, color="teal", label="Aperiodic - Knee mode")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel("Frequency", fontsize=22)
    plt.ylabel("Power", fontsize=22)
    plt.legend(fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(save_path, "fooof_psds.png"), dpi=600, bbox_inches="tight")
    plt.close()

    return None


def compute_fittness_metric(paths):
    """
    Returns errors, R2s and number of peaks in fooof models
    """

    errors, r2s, n_peaks = [], [], []

    # looping through all participants
    for counter, path in enumerate(paths):

        with open(path, "rb") as file:
            fooofres = pickle.load(file=file)

        # looping through channels
        for i in range(fooofres[1].shape[0]):
            fm = fooofres[0].get_fooof(ind=i)
            errors.append(fm.error_)
            r2s.append(fm.r_squared_)
            n_peaks.append(fm.n_peaks_)

    return errors, r2s, n_peaks


def plot_ap_fixed_r2(fixed_r2s, knee_r2s, save_path):
    """
    Returns stacked histogram of R2 scors for aperiodic and periodic
    components
    """
    plt.figure(figsize=(6, 7))
    plt.hist(
        fixed_r2s, bins=100, color="#A9561E", alpha=0.6, label="Aperiodic - fixed mode"
    )
    plt.hist(knee_r2s, bins=100, color="teal", alpha=0.6, label="Aperiodic - knee mode")
    plt.xlim((0.85, 1.01))
    plt.xlabel("$R^2$ value", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks([0.86, 0.9, 0.95, 1])
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.savefig(os.path.join(save_path, "R_2_plot.png"), dpi=600, bbox_inches="tight")
    plt.close()

    return None


def plot_log_log_fooof(freqs, psds, fixed_ap, knee_ap, save_path):

    plt.figure(figsize=(6, 7))
    plt.plot(np.log10(freqs), np.log10(psds), color="black", label="Original PSD")
    plt.plot(
        np.log10(freqs),
        np.log10(fixed_ap),
        color="#A9561E",
        label="Aperiodic - fixed mode",
    )
    plt.plot(
        np.log10(freqs), np.log10(knee_ap), color="teal", label="Aperiodic - Knee mode"
    )
    plt.grid()
    plt.xlabel("Log(Frequency)", fontsize=20)
    plt.ylabel("log(Power)", fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.legend(fontsize=16)
    plt.savefig(
        os.path.join(save_path, "loglogfooof.png"), dpi=600, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":

    # Analyses 1 ----------------------------------------------------------------------
    fixed_path = glob.glob(
        "/home/meganorm-mznasrabadi/Data/CamCAN/fooofResultsNoKnee/sub-CC*"
    )
    knee_path = glob.glob("/home/meganorm-mznasrabadi/Data/CamCAN/fooofResults/sub-CC*")
    save_path = "/home/meganorm-mznasrabadi/Results/natureArticle/fooof_figures/"

    (freqs, psds_avg, fixed_periodics_avg, fixed_aperiodics_avg) = (
        averaged_psds_component(fixed_path)
    )

    (_, _, knee_periodics_avg, knee_aperiodics_avg) = averaged_psds_component(knee_path)

    # plot aperiodic, periodic and PSDs (NOT in log log space)
    plot_knee_fixed_compare_psd(
        freqs, psds_avg, fixed_aperiodics_avg, knee_aperiodics_avg, save_path
    )

    # plot R2 scores for fixed and knee mode
    _, fixed_r2s, _ = compute_fittness_metric(fixed_path)
    _, knee_r2s, _ = compute_fittness_metric(knee_path)
    plot_ap_fixed_r2(fixed_r2s, knee_r2s, save_path)

    # plot aperiodic, periodic and PSDs (IN log log space)
    plot_log_log_fooof(
        freqs, psds_avg, fixed_aperiodics_avg, knee_aperiodics_avg, save_path
    )
