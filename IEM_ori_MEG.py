import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_data(subj):
    data = loadmat("MEG_ori/%s_epochs.mat" % subj)
    return data["trn"], data["trng"].squeeze(), data["ts"]

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

    def run_subject(self, subj, permutation_test=False, plot=False):
        trn, trng, ts = load_data(subj)
        basis_set = self.make_basis_set()

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

def n_correct(coeffs, targ_ori, n_trials):
    n = 0
    tmean = coeffs.mean(axis=2)
    for i in range(n_trials):
        if np.argmax(tmean[i]) == targ_ori:
            n += 1
    return n
    

subjlist = ["AK", "DI", "HHy", "HN", "JL", "KA", "MF", "NN", "SoM", "TE", "VA", "YMi"]

def run_all_subjects(n_ori_chans, permutation_test=False, n_p_tests=10):
    IEM = InvertedEncoder(n_ori_chans)
    total_response = np.zeros(n_ori_chans)
    total_trials = 0
    targ_ori = -1
    print("Running Experimental Test")
    exp_accuracy = 0
    for subj in subjlist:
        coeffs, trng_cv, targ_ori = IEM.run_subject(subj, plot=True)
        exp_accuracy += n_correct(coeffs, targ_ori, len(trng_cv))
        tmean = coeffs.mean(axis=2)
        total_response += np.sum(tmean, 0)
        total_trials += len(trng_cv)

    avg_response = total_response / total_trials
    plt.figure(figsize=(4, 8))
    plt.plot(avg_response)
    plt.ylim(0, 0.3)
    plt.savefig("../Figures/IEM/python_files/avg_response.png")
    plt.clf()
    exp_accuracy /= total_trials
    print("accuracy: {:.3f}".format(exp_accuracy))

    print("Running Permutation Tests")
    if permutation_test:
        permutation_response = np.zeros(n_ori_chans)
        extreme_pts = 0
        extreme_acc = 0
        perm_accuracy = 0
        for i in range(n_p_tests):
            print("Permutation %d" % (i + 1))
            total_response = np.zeros(n_ori_chans)
            total_trials = 0
            temp_perm_accuracy = 0
            for subj in subjlist:
                coeffs, trng_cv, targ_ori = IEM.run_subject(subj, permutation_test=True)
                tmean = coeffs.mean(axis=2)
                temp_perm_accuracy += n_correct(coeffs, targ_ori, len(trng_cv))
                total_response += np.sum(tmean, 0)
                total_trials += len(trng_cv)
            temp_perm_accuracy /= total_trials
            print("accuracy: {:.3f}".format(temp_perm_accuracy))
            perm_accuracy += temp_perm_accuracy
            if temp_perm_accuracy > exp_accuracy:
                extreme_acc += 1

            perm_avg_response = total_response / total_trials
            permutation_response += perm_avg_response
            if perm_avg_response[targ_ori] > avg_response[targ_ori]:
                extreme_pts += 1
        
        permutation_response = permutation_response / n_p_tests
        perm_accuracy /= n_p_tests
        print("accuracy: {:.3f}".format(perm_accuracy))
        plt.figure(figsize=(8, 8))
        plt.bar([1, 2], [exp_accuracy, perm_accuracy], tick_label=["Experimental", "Permutation"])
        plt.title("p-value: {:.3f}".format(extreme_acc / n_p_tests))
        plt.savefig("../Figures/IEM/python_files/accuracy.png")
        plt.clf()

        plt.figure(figsize=(4, 8))
        plt.plot(permutation_response)
        plt.ylim(0, 0.3)
        plt.title("p-value: {:.3f}".format(extreme_pts / n_p_tests))
        plt.savefig("../Figures/IEM/python_files/perm_response.png")
        plt.clf()

def main():
    run_all_subjects(9, permutation_test=True)

if __name__ == "__main__":
    main()
