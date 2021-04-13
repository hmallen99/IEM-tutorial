import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

new_beh_lst = {
    "KA": 1,
    "MF": 7,
    "MK": 12,
    "NNo": 19,
    "KO": 4,
    "HHy": 18,
    "HO": 6,
    "AK": 5,
    "HN": 21,
    "NN": 3,
    "JL": 9,
    "DI": 16,
    "SoM": 2,
    "TE": 17,
    "VA": 10,
    "RS": 14,
    "YMi": 11,
}

def load_data(subj):
    data = loadmat("MEG_ori/%s_epochs2.mat" % subj)
    return data["trn"], data["trng"].squeeze(), data["ts"]

def load_behavior(subj):
    data = loadmat("Behavior/Sub%d_beh.mat" % new_beh_lst[subj])
    return data['TgtOrs'].squeeze(), data['ResOrs'].squeeze()

def calc_relative_orientation(x):
    if np.abs(x) > 90:
        x = x - (np.sign(x) * 180)
        return x
    return x


def load_percept_data(subj, n_classes=9):
    _, trng = load_behavior(subj)
    trng = trng.squeeze()
    trn, _, ts = load_data(subj)
    
    trng = trng[:500]
    trn = trn[~np.isnan(trng)]
    trng = trng[~np.isnan(trng)]
    trng[np.abs(trng) > 90] = trng[np.abs(trng) > 90] - 180 * np.sign(trng[np.abs(trng) > 90])
    offset = 180 / (n_classes * 2)
    trng = (trng + 90 + offset) % 180
    trng = np.floor(trng / (180 / n_classes))
    trng = np.minimum(trng, n_classes-1)
    trng = trng * (180 / n_classes)


    return trn, trng, ts

class InvertedEncoder(object):
    def __init__(self, n_ori_chans):
        self.n_ori_chans = n_ori_chans
        self.chan_center = np.linspace(180 / n_ori_chans, 180, n_ori_chans)
        self.xx = np.linspace(1, 180, 180)


    def make_basis_set(self):
        make_basis_function = lambda xx, mu: np.power(np.cos(np.deg2rad(xx - mu)), self.n_ori_chans - (self.n_ori_chans % 2))
        basis_set = np.zeros((180, self.n_ori_chans))
        for cc in range(self.n_ori_chans):
            basis_set[:, cc] = make_basis_function(self.xx, self.chan_center[cc])
        return basis_set

    def make_stimulus_mask(self, trng):
        stim_mask = np.zeros((len(trng), len(self.xx)))
        for tt in range(stim_mask.shape[0]):
            stim_mask[tt, trng[tt]-1] = 1
        return stim_mask

    def make_trn_repnum(self, trng):
        trn_ou = np.unique(trng)
        trn_repnum = np.zeros(len(trng))
        trn_repnum[:] = np.nan
        n_trials_per_orientation = np.zeros(len(trn_ou))

        for ii in range(len(trn_ou)):
            n_trials_per_orientation[ii] = len(trn_repnum[trng == trn_ou[ii]])
            trn_repnum[trng == trn_ou[ii]] = np.arange(n_trials_per_orientation[ii])

        trn_repnum[trn_repnum >= np.min(n_trials_per_orientation)] = np.nan
        return trn_repnum

    

    def cross_validate(self, trnX_cv, trn_cv, trng_cv, trn_repnum):
        trn_cv_coeffs = np.zeros((len(trng_cv), 2 * trn_cv.shape[1], trn_cv.shape[2]))
        trn_cv_coeffs[:, :trn_cv.shape[1], :] = np.real(trn_cv)
        trn_cv_coeffs[:, trn_cv.shape[1]:, :] = np.imag(trn_cv)
    

        chan_resp_cv_coeffs = np.zeros((trn_cv_coeffs.shape[0], len(self.chan_center), trn_cv_coeffs.shape[2]))

        n_reps = int(np.max(trn_repnum))

        for ii in range(n_reps):
            trnidx = trn_repnum != ii
            tstidx = trn_repnum == ii

            thistrn = trn_cv_coeffs[trnidx, :, :]
            thistst = trn_cv_coeffs[tstidx, :, :]

            for tt in range(thistrn.shape[2]):
                thistrn_tpt = thistrn[:, :, tt]
                thistst_tpt = thistst[:, :, tt]

                w_coeffs = np.linalg.lstsq(trnX_cv[trnidx, :], thistrn_tpt)[0]

                chan_resp_cv_coeffs[tstidx, :, tt] = np.linalg.lstsq(w_coeffs.T, thistst_tpt.T)[0].T

        return chan_resp_cv_coeffs

    def run_subject(self, subj, permutation_test=False, shuffle_data=False, plot=False, percept_data=False, shuffle_idx=[], time_shift=0):
        trn, trng, ts = load_percept_data(subj, self.n_ori_chans) if percept_data else load_data(subj)
        basis_set = self.make_basis_set()

        if time_shift == -1:
            trng[1:len(trng)] = trng[:len(trng)-1]
            trng[0] = np.random.choice(trng)

        if shuffle_data:
            if len(shuffle_idx) == 0:
                shuffle_idx = np.random.choice(len(trng), len(trng), replace=False)
            trn = trn[shuffle_idx]
            trng = trng[shuffle_idx]    

        trng = trng % 180
        trng[trng == 0] = 180
        trng = trng.astype(int)
        trn_repnum = self.make_trn_repnum(trng)
        trng_cv = trng[~np.isnan(trn_repnum)]
        trn_cv = trn[~np.isnan(trn_repnum)]

        if permutation_test:
            np.random.shuffle(trng_cv)

        stim_mask = self.make_stimulus_mask(trng_cv)
        trnX_cv = stim_mask @ basis_set

        trn_repnum = trn_repnum[~np.isnan(trn_repnum)]

        coeffs = self.cross_validate(trnX_cv, trn_cv, trng_cv, trn_repnum)

        targ_ori = int(np.round(len(self.chan_center) / 2))
        coeffs_shift = np.zeros(coeffs.shape)
        for ii in range(self.n_ori_chans):
            idx = trng_cv == self.chan_center[ii]
            coeffs_shift[idx, :, :] = np.roll(coeffs[idx, :, :], targ_ori - ii, axis=1)

        if plot:
            tmean = coeffs_shift.mean(axis=2)

            plt.figure(figsize=(4, 8))
            plt.plot(tmean.mean(axis=0))
            plt.ylim(0, 0.3)
            plt.savefig("../Figures/IEM/python_files/%s_response.png" % subj)
            plt.clf()

        return coeffs_shift, trng_cv, targ_ori

    def get_diffs(self, subj):
        tgt, _ = load_behavior(subj)
        tgt = tgt[:500]
        diffs = [0]
        for i in range(1, len(tgt)):
            diffs.append(calc_relative_orientation(tgt[i - 1] - tgt[i]))
        diffs = np.array(diffs)
        return diffs

    def run_sd_subject(self, subj, n_bins=30, permutation_test=False):
        _, trng, _ = load_data(subj)
        trn_repnum = self.make_trn_repnum(trng)
        diffs = self.get_diffs(subj)
        diffs = diffs[~np.isnan(trn_repnum)]
        if permutation_test:
            np.random.shuffle(diffs)
        shuffle_idx = np.random.choice(len(trng), len(trng), replace=False)
        coeffs_shift, trng_cv, targ_ori = self.run_subject(subj, shuffle_data=True, shuffle_idx=shuffle_idx)
        bins = np.zeros((n_bins, self.n_ori_chans, coeffs_shift.shape[2]))
        bin_sizes = np.zeros(n_bins)
        diffs = np.minimum(np.floor((diffs + 90) / (180 / n_bins)), n_bins - 1).astype(int)
        for i in range(len(trng_cv)):
            bins[diffs[i]] += coeffs_shift[i]
            bin_sizes[diffs[i]] += 1
        
        return bins, bin_sizes
            


def n_correct(tmean, targ_ori, n_trials):
    n = 0
    for i in range(n_trials):
        if np.argmax(tmean[i]) == targ_ori:
            n += 1
    return n
    
def n_correct_tsteps(coeffs, targ_ori, n_trials):
    n_timesteps = coeffs.shape[2]
    n = np.zeros(n_timesteps)
    for i in range(n_trials):
        for j in range(n_timesteps):
            if np.argmax(coeffs[i, :, j]) == targ_ori:
                n[j] += 1
    return n

subjlist = ["AK", "DI", "HHy", "HN", "JL", "KA", "MF", "NN", "SoM", "TE", "VA", "YMi"]

def make_pd_bar(exp_accs, perm_accs):
    data_lst = []
    data_lst.extend(exp_accs)
    data_lst.extend(perm_accs)

    str_lst = []
    str_lst.extend(["experimental" for _ in exp_accs])
    str_lst.extend(["permutation" for _ in perm_accs])

    d = {'accuracy': data_lst, 'Trial': str_lst}
    df = pd.DataFrame(data=d)
    return df

def run_all_subjects(n_ori_chans, n_p_tests=100, n_exp_tests=10, n_timesteps=16, percept_data=False, time_shift=0):
    IEM = InvertedEncoder(n_ori_chans)
    avg_response = np.zeros(n_ori_chans)
    total_trials = 0
    targ_ori = -1

    print("Running Experimental Test")
    exp_list_x = []
    exp_list_y = []
    exp_accuracies = np.zeros(n_exp_tests)
    exp_t_accs_x = []
    exp_t_accs_y = []
    exp_ch_responses = np.zeros((n_exp_tests, n_ori_chans))
    for i in range(n_exp_tests):
        print("Trial %d" % (i + 1))
        test_trials = 0
        trial_response = np.zeros(n_ori_chans)
        trial_accuracy = np.zeros(n_timesteps)
        for subj in subjlist:
            coeffs, trng_cv, targ_ori = IEM.run_subject(subj, shuffle_data=True, 
                                                        percept_data=percept_data, time_shift=0)    
            tmean = coeffs.mean(axis=2)
            exp_accuracies[i] += n_correct(tmean, targ_ori, len(trng_cv))
            trial_accuracy += n_correct_tsteps(coeffs, targ_ori, len(trng_cv))
            trial_response += np.sum(tmean, 0)
            total_trials += len(trng_cv)
            test_trials += len(trng_cv)
        exp_accuracies[i] /= test_trials

        trial_accuracy = trial_accuracy / test_trials
        exp_t_accs_x.extend(np.linspace(0, 0.375, n_timesteps))
        exp_t_accs_y.extend(trial_accuracy)

        avg_response += trial_response
        trial_response = trial_response / test_trials
        exp_ch_responses[i] += trial_response
        exp_list_x.extend(np.arange(n_ori_chans))
        exp_list_y.extend(trial_response)
        print("Test Accuracy: {:.3f}".format(exp_accuracies[i]))

    avg_response = avg_response / total_trials

    print("Running Permutation Tests")
    permutation_response = np.zeros(n_ori_chans)
    extreme_accs = np.zeros(n_exp_tests)
    perm_t_accs_x = []
    perm_t_accs_y = []
    perm_accuracy = 0
    perm_list_x = []
    perm_list_y = []
    perm_accuracies = []
    extreme_ch_resps = np.zeros(n_ori_chans)
    for i in range(n_p_tests):
        print("Permutation %d" % (i + 1))
        total_response = np.zeros(n_ori_chans)
        total_trials = 0
        temp_perm_accuracy = 0
        perm_trial_acc = np.zeros(n_timesteps)

        for subj in subjlist:
            coeffs, trng_cv, targ_ori = IEM.run_subject(subj, shuffle_data=True, permutation_test=True, 
                                                        percept_data=percept_data, time_shift=time_shift)
            tmean = coeffs.mean(axis=2)
            temp_perm_accuracy += n_correct(tmean, targ_ori, len(trng_cv))
            perm_trial_acc += n_correct_tsteps(coeffs, targ_ori, len(trng_cv))
            total_response += np.sum(tmean, 0)
            total_trials += len(trng_cv)
        temp_perm_accuracy /= total_trials
        print("accuracy: {:.3f}".format(temp_perm_accuracy))
        perm_accuracy += temp_perm_accuracy
        for i in range(n_exp_tests):
            if temp_perm_accuracy >= exp_accuracies[i]:
                extreme_accs[i] += 1
        perm_accuracies.append(temp_perm_accuracy)

        perm_trial_acc = perm_trial_acc / total_trials
        perm_t_accs_x.extend(np.linspace(0, 0.375, n_timesteps))
        perm_t_accs_y.extend(perm_trial_acc)

        perm_avg_response = total_response / total_trials
        for i in range(n_ori_chans):
            for j in range(n_exp_tests):
                if perm_avg_response[i] >= exp_ch_responses[j][i]:
                    extreme_ch_resps[i] += 1

        perm_list_x.extend(np.arange(n_ori_chans))
        perm_list_y.extend(perm_avg_response)
        permutation_response += perm_avg_response

    ch_resp_p_values = extreme_ch_resps / (n_exp_tests * n_p_tests)
    acc_df = make_pd_bar(exp_accuracies, perm_accuracies)
    permutation_response = permutation_response / n_p_tests
    perm_accuracy /= n_p_tests
    print("accuracy: {:.3f}".format(perm_accuracy))

    plt.figure(figsize=(8, 8))
    sns.barplot(x='Trial', y='accuracy', data=acc_df, ci="sd")
    plt.title("Mean Accuracy, p-value: {:.3f}".format(extreme_accs.mean() / n_p_tests))
    plt.savefig("../Figures/IEM/python_files/accuracy.png")
    plt.clf()

    plt.figure(figsize=(8,8))
    plt.hist(perm_accuracies, bins=np.linspace(0.09, 0.14, 20), label="permutation")
    plt.axvline(exp_accuracies.mean(), label="Mean Accuracy", linewidth=2.5, color='r')
    plt.title("Accuracy vs. Null Distribution")
    plt.legend()
    plt.xlabel("Accuracy")
    plt.ylabel("Occurences")
    plt.savefig("../Figures/IEM/python_files/null_comparison.png")
    plt.clf()

    plt.figure(figsize=(4, 8))
    sns.lineplot(exp_list_x, exp_list_y, label="Experimental Response", ci="sd")
    sns.lineplot(perm_list_x, perm_list_y, label="Permutation", ci="sd")
    
    plt.ylim(0.1, 0.35)
    for i in range(n_ori_chans):
        plt.text(i-0.5, avg_response[i]+0.03, ch_resp_p_values[i], color="red")
    plt.legend()
    plt.xticks(ticks=np.arange(n_ori_chans), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
    plt.title("Mean Channel Response of Experiment vs. Permutation groups")
    plt.xlabel("Orientation Channel (Degrees)")
    plt.ylabel("Channel Response")
    plt.savefig("../Figures/IEM/python_files/perm_response.png")
    plt.clf()

    plt.figure(figsize=(8, 8))
    sns.lineplot(exp_t_accs_x, exp_t_accs_y, label="Experimental Accuracy", ci="sd")
    sns.lineplot(perm_t_accs_x, perm_t_accs_y, label="Permutation Accuracy", ci="sd")
    plt.legend()
    plt.ylim(0.05, 0.15)
    plt.title("Accuracy at each timestep")
    plt.savefig("../Figures/IEM/python_files/timestep_accuracy.png")
    plt.clf()



def iem_sd_all(n_ori_chans, n_bins=15, percept_data=False, n_p_tests=100, n_exp_tests=50, n_timesteps=16):
    IEM = InvertedEncoder(n_ori_chans)
    
   
    bin_accuracies = np.zeros((n_exp_tests, n_bins, n_ori_chans, n_timesteps))
    for i in range(n_exp_tests):
        print("Experimental Test: %i" % i)
        bin_sizes = np.zeros(n_bins)
        bins = np.zeros((n_bins, n_ori_chans, n_timesteps))
        for subj in subjlist:
            temp_bins, temp_bin_sizes = IEM.run_sd_subject(subj, n_bins=n_bins)
            bins += temp_bins
            bin_sizes += temp_bin_sizes
        
        for j in range(n_timesteps):
            bin_accuracies[i, :, :, j] = bins[:, :, j] / bin_sizes[:, None]

    perm_bin_accuracies = np.zeros((n_p_tests, n_bins, n_ori_chans, n_timesteps))
    for i in range(n_p_tests):
        print("Permutation Test: %i" %i)
        bin_sizes = np.zeros(n_bins)
        bins = np.zeros((n_bins, n_ori_chans, n_timesteps))
        for subj in subjlist:
            temp_bins, temp_bin_sizes = IEM.run_sd_subject(subj, n_bins=n_bins, permutation_test=True)
            bins += temp_bins
            bin_sizes += temp_bin_sizes
        
        for j in range(n_timesteps):
            perm_bin_accuracies[i, :, :, j] = bins[:, :, j] / bin_sizes[:, None]
        
    for j in range(n_timesteps):
        figure = plt.figure(figsize=(8,10))
        ax0 = figure.add_subplot(2, 1, 1)
        im0 = ax0.imshow(bin_accuracies[:, :, :, j].mean(axis=0).T, aspect="equal")
        ax0.set_xlabel("Relative Previous Orientation")
        figure.colorbar(im0, ax=ax0)
        plt.xticks(ticks=np.arange(-0.5, n_bins+0.5, 1), labels=np.linspace(-90, 90, n_bins+1).astype(int))
        plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
        plt.ylabel("Channel response")
        plt.title("Channel response binned by previous orientation, t=%d" % j)

        ax1 = figure.add_subplot(2, 1, 2)
        im1 = ax1.imshow(perm_bin_accuracies[:, :, :, j].mean(axis=0).T, aspect="equal")
        ax1.set_xlabel("Relative Previous Orientation")
        figure.colorbar(im1, ax=ax1)

        plt.xticks(ticks=np.arange(-0.5, n_bins+0.5, 1), labels=np.linspace(-90, 90, n_bins+1).astype(int))
        plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
        plt.title("Permutation")
        
        
        plt.ylabel("Channel response")
        plt.savefig("../Figures/IEM/python_files/sd_all_t%d.png" % j)
        plt.clf()
    return

def main():
    #run_all_subjects(9, percept_data=False)
    #load_behavior("KA")
    #load_percept_data("KA")
    iem_sd_all(9)

if __name__ == "__main__":
    main()
