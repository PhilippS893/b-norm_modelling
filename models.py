import os, pickle
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from pcntoolkit.normative import estimate, predict, evaluate, compute_MSLL
from pcntoolkit.util.utils import create_design_matrix
from utils import calibration_descriptives


class BaseModel(ABC):
    """Base normative model functionality.
    """

    def __init__(self, rois: str):
        self.rois = rois

    def set_output_directories(self, project_dir: str, target_dir: str):
        self.model_output_dir = os.path.join(project_dir, "model_output", target_dir, self.__class__.__name__)
        os.makedirs(self.model_output_dir, exist_ok=True)

        self.data_dir = os.path.join(project_dir, "data", target_dir, self.__class__.__name__)
        os.makedirs(self.data_dir, exist_ok=True)

        self.plot_dir = os.path.join(project_dir, "plots", target_dir, self.__class__.__name__)
        os.makedirs(self.plot_dir, exist_ok=True)

    @abstractmethod
    def setup_dummy_data(self, xmin: int, xmax: int, sex: int, site_list: list,
                         n_dummies: int = None, site_ids: list = None, model_type: str = 'full'):
        pass

    def load(self, path):
        if not os.path.exists(os.path.join(path, "Models")):
            FileNotFoundError()

        path_to_model = os.path.join(path, "Models")

        with open(os.path.join(path_to_model, "NM_0_0_estimate.pkl"), "rb") as handle:
            self.nm = pickle.load(handle)

        self.W = self.nm.blr.warp
        self.warp_param = self.nm.blr.hyp[1:self.nm.blr.warp.get_n_params() + 1]

    def train(self, train_dir, train_tpt, test_dir, test_tpt, data_split: str = "full", suffix: str = "_w4yr",
              use_bspline=True, **kwargs):
        """
        Function to train the normative model.

        Parameters
        ----------
        train_dir : str/path
            directory for training files
        test_dir : str/path
            directory for testing files
        timepoint : str
            which timepoint to use (bl, 2yr, or 4yr)
        data_split : str, Default: "full"
            which dataset to use (full, male, or female)
        suffix : str, Default: "_w4yr"
            what to append to the filename
        use_bspline : bool, Default: True
            use bsplines for training

        kwargs
        ------
        project_dir : str/path, Default: os.getcwd()
            path to project
        warp : str, Default: 'WarpSinArcsinh'
            which warping algorithm to use
        do_cv : bool, Default: False
            should cross-validation be performed
        cv_fold : int, Default: None
            how many cross-validation folds to run.
        """
        # parse keyword arguments
        project_dir = kwargs.pop('project_dir', os.getcwd())
        warp = kwargs.pop('warp', 'WarpSinArcsinh')
        do_cv = kwargs.pop('do_cv', False)
        cv_fold = kwargs.pop('cv_fold', None)

        bspline_suffix = '_bspline' if use_bspline else ''

        current_model_dir = f"{self.model_output_dir}{bspline_suffix}"

        # make sure, we are in the specified project_dir
        os.chdir(project_dir)

        blr_metrics = pd.DataFrame(columns=["ROI", "MSLL", "EV", "SMSE", "RMSE", "Rho", "BIC"])
        if do_cv:
            blr_metrics = pd.DataFrame(columns=["ROI", "MSLL", "EV", "SMSE", "RMSE", "Rho", "BIC", "fold"])

        for i, roi in enumerate(self.rois):
            print("TRAINING ROI: ", roi)

            roi_dir = os.path.join(current_model_dir, data_split, roi)
            if do_cv:
                roi_dir = os.path.join(roi_dir, "CV", f"fold{cv_fold}")

            os.makedirs(roi_dir, exist_ok=True)
            os.chdir(roi_dir)

            train_cov = os.path.join(self.data_dir, train_dir, "cov_files_train", f"{roi}_{train_tpt}{bspline_suffix}.txt")
            train_resp = os.path.join(self.data_dir, train_dir, "resp_files_train", f"{roi}_{train_tpt}.txt")
            test_cov = os.path.join(self.data_dir, test_dir, "cov_files_test", f"{roi}_{test_tpt}{bspline_suffix}.txt")
            test_resp = os.path.join(self.data_dir, test_dir, "resp_files_test", f"{roi}_{test_tpt}.txt")

            # train the model using all data
            estimate(train_cov, train_resp, testresp=test_resp, testcov=test_cov, alg='blr',
                     optimizer='powell', savemodel=True, standardize=False, warp=warp, warp_reparam=True)

            os.makedirs(roi_dir + f'/evaluate_{test_tpt}{suffix}', exist_ok=True)
            os.chdir(roi_dir + f'/evaluate_{test_tpt}{suffix}')

            print(os.path.join(roi_dir, "Models"))
            _, _, _ = predict(test_cov, test_resp, alg='blr', model_path=os.path.join(roi_dir, 'Models'))
            os.chdir(roi_dir)

            # load training data (required for MSLL)
            y_tr = np.loadtxt(train_resp)
            y_tr = y_tr[:, np.newaxis]

            # load test data, compute metrics on whole test set, save to pandas df
            # TEST THE TRAINING FILE
            y_te = np.loadtxt(test_resp)

            y_te = y_te[:, np.newaxis]
            yhat_te = np.loadtxt(os.path.join(roi_dir, 'yhat_estimate.txt'))
            s2_te = np.loadtxt(os.path.join(roi_dir, 'ys2_estimate.txt'))
            yhat_te = yhat_te[:, np.newaxis]
            s2_te = s2_te[:, np.newaxis]

            if warp is None:
                metrics_te = evaluate(y_te, yhat_te, s2_te)
                y_mean_te = np.array([[np.mean(y_te)]])
                y_var_te = np.array([[np.var(y_te)]])
                MSLL_te = compute_MSLL(y_te, yhat_te, s2_te, y_mean_te, y_var_te)
            else:
                self.load(".")
                # warp predictions
                med_te = self.W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), self.warp_param)[0]
                med_te = med_te[:, np.newaxis]

                # evaluation metrics
                metrics_te = evaluate(y_te, med_te)

                # compute MSLL manually
                y_te_w = self.W.f(y_te, self.warp_param)
                y_tr_w = self.W.f(y_tr, self.warp_param)
                y_tr_mean = np.array([[np.mean(y_tr_w)]])
                y_tr_var = np.array([[np.var(y_tr_w)]])
                MSLL_te = compute_MSLL(y_te_w, yhat_te, s2_te, y_tr_mean, y_tr_var)

            BIC = len(self.nm.blr.hyp) * np.log(y_tr.shape[0]) + 2 * self.nm.neg_log_lik
            if do_cv:
                blr_metrics.loc[len(blr_metrics)] = [roi, MSLL_te[0], metrics_te['EXPV'][0], metrics_te['SMSE'][0],
                                                     metrics_te['RMSE'][0],
                                                     metrics_te['Rho'][0], BIC, cv_fold]
            else:
                blr_metrics.loc[len(blr_metrics)] = [roi, MSLL_te[0], metrics_te['EXPV'][0], metrics_te['SMSE'][0],
                                                     metrics_te['RMSE'][0],
                                                     metrics_te['Rho'][0], BIC]

        # make sure we're back in the project directory
        os.chdir(project_dir)
        if do_cv:
            blr_metrics.to_csv(os.path.join(current_model_dir, data_split, f"cv_performance_fold{cv_fold}.csv"),
                               index=False)
        else:
            blr_metrics.to_csv(os.path.join(current_model_dir, data_split, f"{test_tpt}_performance{suffix}.csv"),
                               index=False)

    def evaluate(self, test_dir, test_tpt, out_name, do_cross=False, data_split: str = "full", use_bspline=True,
                 **kwargs):
        """
        Function to evaluate the normative model.

        Parameters
        ----------
        test_dir : str/path
            directory for testing files
        test_tpt : str
            which timepoint to use (bl, 2yr, or 4yr)
        out_name : str
            where to save the model etc.
        data_split : str, Default: "full"
            which data to use (full, male, or female)
        use_bspline : bool, Default: True
            use bsplines for training

        kwargs
        ------
        project_dir : str/path, Default: os.getcwd()
            path to project
        """
        # parse keyword arguments
        project_dir = kwargs.pop('project_dir', os.getcwd())

        #  make sure we are in the project directory
        os.chdir(project_dir)

        # predefine the blr_metric dataframe
        blr_metrics = pd.DataFrame(columns=['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'BIC', 'skew', 'kurtosis'])

        cross_ds = None
        if do_cross and data_split != 'full':
            cross_ds = 'female' if data_split == 'male' else 'male'

        bspline_suffix = '_bspline' if use_bspline else ''
        current_model_dir = f"{self.model_output_dir}{bspline_suffix}"

        for i, roi in enumerate(self.rois):
            print('EVALUATING ROI: ', roi, 'for timepoint ', test_tpt)
            roi_dir = os.path.join(current_model_dir, data_split, roi)

            if do_cross:
                eval_dir = os.path.join(roi_dir, f'evaluate_{test_tpt}_cross')
            else:
                eval_dir = os.path.join(roi_dir, f'evaluate_{test_tpt}')

            os.makedirs(eval_dir, exist_ok=True)
            os.chdir(eval_dir)

            # load the covariate & resp file
            if do_cross:
                inter = os.path.join(self.data_dir, cross_ds)
                cov_file_test = os.path.join(inter, "cov_files_test", f"{roi}_{test_tpt}{bspline_suffix}.txt")
                resp_file_test = os.path.join(inter, "resp_files_test", f"{roi}_{test_tpt}.txt")
            else:
                inter = os.path.join(self.data_dir, test_dir)
                cov_file_test = os.path.join(inter, "cov_files_test", f"{roi}_{test_tpt}{bspline_suffix}.txt")
                resp_file_test = os.path.join(inter, "resp_files_test", f"{roi}_{test_tpt}.txt")

            # set the respective model path
            if data_split in ["full_male", "full_female"]:
                model_path = os.path.join(current_model_dir, "full", roi)
            else:
                model_path = os.path.join(current_model_dir, data_split, roi)

            # do the prediction
            yhat_te, s2_te, z = predict(cov_file_test, resp_file_test, alg='blr',
                                        model_path=os.path.join(model_path, "Models"))

            # evaluate
            y_te = np.loadtxt(resp_file_test)
            y_te = y_te[:, np.newaxis]

            self.load(model_path)

            if self.W is None:
                metrics_te = evaluate(y_te, yhat_te, s2_te)
                y_mean_te = np.array([[np.mean(y_te)]])
                y_var_te = np.array([[np.var(y_te)]])
                MSLL_te = compute_MSLL(y_te, yhat_te, s2_te, y_mean_te, y_var_te)

            else:
                # warp predictions
                med_te = self.W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), self.warp_param)[0]
                med_te = med_te[:, np.newaxis]

                # evaluation metrics
                y_mean_te = np.array([[np.mean(y_te)]])
                y_var_te = np.array([[np.var(y_te)]])

                metrics_te = evaluate(y_te, med_te)
                MSLL_te = compute_MSLL(y_te, yhat_te, s2_te, y_mean_te, y_var_te)

            BIC = len(self.nm.blr.hyp) * np.log(y_te.shape[0]) + 2 * self.nm.neg_log_lik
            # compute skew and kurtosis for the distribution of the Z-scores
            Z = np.loadtxt(os.path.join(eval_dir, 'Z_predict.txt'))
            [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
            blr_metrics.loc[len(blr_metrics)] = [roi, MSLL_te[0], metrics_te['EXPV'][0], metrics_te['SMSE'][0],
                                                 metrics_te['RMSE'][0], metrics_te['Rho'][0], BIC, skew, kurtosis]

        blr_metrics.to_csv(os.path.join(current_model_dir, data_split, out_name), index=False)
        os.chdir(project_dir)


class CNorm(BaseModel):
    """
    cross-sectional normative age model
    """

    def setup_dummy_data(self, xmin: int, xmax: int, sex: int, site_list: list,
                         n_dummies: int = None, site_ids: list = None, bl_age=None, model_type: str = 'full'):
        self.covariateColumns = ["age", "sex"]
        xx = np.arange(xmin, xmax, 1)
        X0_dummy = np.zeros((len(xx), len(self.covariateColumns)))
        X0_dummy[:, 0] = xx
        X0_dummy[:, 1] = sex

        if model_type == 'sexspecific':
            self.covariateColumns = ["age"]
            X0_dummy = np.expand_dims(X0_dummy[:, 0], axis=1)

        self.X_dummy_bspline = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax,
                                                    site_ids=site_ids, all_sites=site_list)
        cov_file_dummy = f'cov_dummy_mean_sex{sex}_bspline.txt'
        np.savetxt(cov_file_dummy, self.X_dummy_bspline)

        self.X_dummy = self.X_dummy_bspline[:, :(1 + len(self.covariateColumns) + len(site_list))]
        cov_file_dummy = f'cov_dummy_mean_sex{sex}.txt'
        np.savetxt(cov_file_dummy, self.X_dummy)


class CT(BaseModel):
    """
    Thickness only model
    """
    def setup_dummy_data(self, xmin: int, xmax: int, sex: int, site_list: list,
                         n_dummies: int = None, site_ids: list = None, bl_age=None):
        self.covariateColumns = ["thickness"]
        xx = np.arange(xmin, xmax, (xmax - xmin) / n_dummies)
        X0_dummy = np.zeros((len(xx), 1))
        X0_dummy[:, 0] = xx

        self.X_dummy_bspline = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax,
                                                    site_ids=site_ids, all_sites=site_list)

        cov_file_dummy = f'cov_dummy_mean_sex{sex}_bspline.txt'
        np.savetxt(cov_file_dummy, self.X_dummy_bspline)

        self.X_dummy = self.X_dummy_bspline[:, :(2 + len(site_list))]
        cov_file_dummy = f'cov_dummy_mean_sex{sex}.txt'
        np.savetxt(cov_file_dummy, self.X_dummy)


class BNorm(BaseModel):
    """
    baseline-integrated normative model.
    """

    def setup_dummy_data(self, xmin: int, xmax: int, sex: int, site_list: list,
                         n_dummies: int = None, site_ids: list = None, bl_age=None, model_type='full'):
        self.covariateColumns = ["thickness", "age_bl", "age_tx", "sex"]
        xx = np.arange(xmin, xmax, (xmax - xmin) / n_dummies)
        xx = xx[:n_dummies]

        X0_dummy = np.zeros((n_dummies, len(self.covariateColumns)))
        X0_dummy[:n_dummies, 0] = xx
        X0_dummy[:, 1] = bl_age
        X0_dummy[:, 2] = X0_dummy[:, 1] + 24
        self.age_4yr_dummy = X0_dummy[:, 1] + 48

        if model_type == 'sexspecific':
            self.covariateColumns = self.covariateColumns[:-1]
            X0_dummy = X0_dummy[:, :-1]

        self.X_dummy_bspline = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax,
                                                    site_ids=site_ids, all_sites=site_list)

        cov_file_dummy = f'cov_dummy_mean_sex{sex}_blage{bl_age}_bspline.txt'
        np.savetxt(cov_file_dummy, self.X_dummy_bspline)

        self.X_dummy = self.X_dummy_bspline[:, :(1 + len(self.covariateColumns) + len(site_list))]
        cov_file_dummy = f'cov_dummy_mean_sex{sex}_blage{bl_age}.txt'
        np.savetxt(cov_file_dummy, self.X_dummy)
