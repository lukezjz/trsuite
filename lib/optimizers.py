import numpy as np
from lib import utils


class SimulatedAnnealing:
    def __init__(self, schedule, aa_probabilities_L, cst_bias):
        self.T0, self.n_steps, self.decrease_factor, self.decrease_range, self.n_muts = schedule
        self.aa_probabilities_L = aa_probabilities_L
        self.L = len(aa_probabilities_L)
        self.idx_allowed = np.arange(self.L)[np.any((aa_probabilities_L != 0.0) & (aa_probabilities_L != 1.0), axis=1)]
        self.aa_idx = np.arange(20)
        self.cst_bias = cst_bias

    def iterator(self, starting_msa, loss_function, output, nsave=50):
        msa = starting_msa.copy()
        losses = loss_function(msa)
        T = self.T0
        if "constraints_loss" in losses:
            cst_loss = losses["constraints_loss"] * (1 + self.cst_bias)

        for i in range(self.n_steps):
            marker = True
            while marker:
                current_msa = self.sampler(msa)
                current_losses = loss_function(current_msa)

                if "constraints_loss" in losses and losses["constraints_loss"] > cst_loss:
                    print(losses["constraints_loss"], cst_loss)
                    continue
                else:
                    marker = False

                if current_losses["total_loss"] < losses["total_loss"]:
                    msa = current_msa
                    losses = current_losses
                else:
                    if np.exp((losses["total_loss"] - current_losses["total_loss"]) / T) > np.random.uniform():
                        msa = current_msa
                        losses = current_losses

                if i % nsave == 0:
                    loss_line = ", ".join(["{}={:<6f}".format(name, losses[name]) for name in sorted(current_losses.keys())])
                    content = "{:<5d}, {:<5f}, {}, {}\n".format(i, T, utils.seq_idx2aa(msa[0]), loss_line)
                    print(content[:-1])
                    with open(output, "a") as fa:
                        fa.write(content)

                if i % self.decrease_range == 0 and i != 0:
                    T /= self.decrease_factor

        return msa

    def sampler(self, msa):
        current_msa = msa.copy()
        for i in range(self.n_muts):
            idx = np.random.choice(self.idx_allowed)
            current_msa[0, idx] = np.random.choice(self.aa_idx, p=self.aa_probabilities_L[idx])
        return current_msa
