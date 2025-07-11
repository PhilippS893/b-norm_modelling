import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Plotter:

    def __init__(self, model):
        self.model = model

    def plot_performances(self, data_splits: list, file_names: list = None, horizontal: bool = False,
                          figsize: tuple = (30, 5), file_type: str = "png", use_bspline: bool = False):
        """
        Function to plot box plots of performance values obtained from training the normative models.

        Parameters
        ----------
        data_splits : list
            what should be plotted (full, male, female)
        file_names : list, Default: None
            file names containing performance values. If None, use standard names
        use_bspline : bool, Default: True
            use bsplines data
        """
        if file_names is None:
            file_names = ['t0_performance_2yr.csv', 't1_performance_2yr.csv', 't2_performance_4yr.csv']

        bspline_suffix = '_bspline' if use_bspline else ''

        dfs = []

        for d, ds in enumerate(data_splits):
            for i in range(len(file_names)):
                df = pd.read_csv(os.path.join(f"{self.model.model_output_dir}{bspline_suffix}", ds, file_names[i]))
                df['timepoint'] = file_names[i].split("_")[0]
                df['data_split'] = ds
                dfs.append(df)

        combined = pd.concat(dfs, axis=0).reset_index(drop=True)

        metrics = ['SMSE', 'EV', 'Rho', 'BIC', 'skew', 'kurtosis']
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        for i, m in enumerate(metrics):
            if horizontal:
                g = sns.boxplot(data=combined, y='data_split', x=m, hue='timepoint', ax=axes[i],
                                flierprops=dict(marker='+'))
            else:
                g = sns.boxplot(data=combined, x='data_split', y=m, hue='timepoint', ax=axes[i],
                                flierprops=dict(marker='+'))
            if i > 0:
                g.legend_.remove()

        fig.tight_layout()
        plt.savefig(os.path.join(self.model.plot_dir, f'performance_measures.{file_type}'), transparent=True)

        return fig, axes


    def add_training_data(self, axes, dataset: str, roi_label: str, idcs, med,
                          site_list: list, timepoint: str = "2yr", use_bspline: bool=False, 
                          color="purple", marker="s", alpha=.3):
        """
        Adds training data to normative plots

        Parameters
        ----------
        axes : matplotlib.pyplot.axes
            axes on which the training data should be added to.
        """

        bspline_suffix = '_bspline' if use_bspline else ''
        data_dir = os.path.join(self.model.data_dir + bspline_suffix, dataset, '2yr_only')

        X_tr = np.loadtxt(os.path.join(data_dir, 'cov_files_train', f'{roi_label}_{timepoint}{bspline_suffix}.txt'))
        y_tr = np.loadtxt(os.path.join(data_dir, 'resp_files_train', f'{roi_label}_{timepoint}.txt'))

        X_rows2use_train = X_tr[idcs, :]
        y_to_use_train = y_tr[idcs]

        legend_ctr = 0
        for sid, site in enumerate(site_list):
            idx = np.where(X_rows2use_train[:, sid + len(self.model.covariateColumns) + 1] != 0)[0]
            if len(idx) == 0:
                continue
            idx_dummy = np.bitwise_and(self.model.X_dummy[:, 1] > X_rows2use_train[idx, 1].min(),
                                       self.model.X_dummy[:, 1] < X_rows2use_train[idx, 1].max())
            y_te_rescaled = y_to_use_train[idx] - np.median(y_to_use_train[idx]) + np.median(med[idx_dummy])
            axes.scatter(X_rows2use_train[idx, 1], y_te_rescaled, s=24, color=color,
                         alpha=alpha, marker=marker, label=f"train" if legend_ctr == 0 else None)
            if legend_ctr == 0:
                legend_ctr += 1


    def add_centile_curves(self, xx, yhat, s2, s2s, ax, percentiles=None, clr='grey') -> None:
        """
        Adds the centile patches to the normative plots
        """
        if percentiles is None:
            percentiles = [[.25, .75], [.05, .95], [.01, .99]]

        W = self.model.W
        warp_param = self.model.warp_param

        for _, p in enumerate(percentiles):
            # fill the gaps in between the centiles
            _, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=p)
            ax.fill_between(xx, pr_int[:, 0], pr_int[:, 1], alpha=0.1, color=clr)

            # make the width of each centile proportional to the epistemic uncertainty
            _, pr_intl = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2 - 0.5 * s2s), warp_param, percentiles=p)
            _, pr_intu = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2 + 0.5 * s2s), warp_param, percentiles=p)
            ax.fill_between(xx, pr_int[:, 0], pr_intu[:, 0], alpha=0.1, color=clr)
            ax.fill_between(xx, pr_intl[:, 1], pr_intu[:, 1], alpha=0.3, color=clr)

            # plot actual centile lines
            ax.plot(xx, pr_int[:, 0], color=clr, linewidth=0.5)
            ax.plot(xx, pr_int[:, 1], color=clr, linewidth=0.5)