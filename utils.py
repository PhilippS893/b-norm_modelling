import os, typing
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import kruskal
from scipy import special as spc
from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import create_bspline_basis, compute_MSLL, create_design_matrix
from sklearn.model_selection import train_test_split


def remove_axes(axes, to_remove: list=["top", "right"]):
    """
    Remove spines from supplied figure axes.
    
    Parameters
    ----------
    axes : matplotlib.axes
    to_remove : list, default: ["top", "right"]
    
    Returns
    -------
    axes : matplotlib.axes where spines in to_remove were set invisible.
    """
    try:
        if len(axes.shape) > 1:
            axes = axes.flatten()
    except:
        axes.spines[to_remove].set_visible(False);
        return

    for i, ax in enumerate(axes):
        ax.spines[to_remove].set_visible(False);


def split_first_col_of_glasser(df: pd.DataFrame, drop_extra: bool=True) -> pd.DataFrame:
    """
    Splits the first column into a ``src_subject_id`` and ``eventname`` column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that contains the concatenated subjectid_eventname in the first column.
    drop_extra : bool, default: True
    
    Returns
    -------
    df : pandas.DataFrame 
        DataFrame where the first column was split into a ``src_subject_id`` and 
        ``eventname`` column.
    """
    id_list, event_list = [], []
    for i, this_str in enumerate(df.iloc[:, 0].values):
        inter = this_str.split("_")
        id_list.append(inter[0])
        event_list.append(inter[1])

    if drop_extra:
        df.drop(columns=[df.columns[0], *df.columns[-3:]], inplace=True)
    else:
        df.drop(columns=[df.columns[0]], inplace=True)

    # insert the two columns at the begining of the DF
    df.insert(0, 'src_subject_id', id_list)
    df.insert(1, 'eventname', event_list)
    df.eventname = df.eventname.str.lower()

    return df


def adjust_sub_and_event_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes underscores from a ``src_subject_id`` and ``eventname`` column.
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    Returns
    -------
    df : pandas.DataFrame 
    """
    df.src_subject_id = df.src_subject_id.str.replace("_", "")
    df.eventname = df.eventname.str.replace("_", "")
        
    return df

def female_categorical(row: pd.DataFrame, care_or_youth: str) -> pd.DataFrame:
    """
    female: body hair growth + breast development || using menarche info as follows:
    prepubertal = 2 + no menarche
    early pubertal = 3 + no menarche
    midpubertal =>3 + no menarche
    late pubertal <=7 + menarche
    postpubertal = 8 + menarche
    according to Herting et al (2021) Frontiers in Endocrinology

    Author: Dominik Kraft
    """
    ### based on caregiver or youth report, menarche variable is labeled differently
    if care_or_youth == "youth":
        menarche = row["pds_f5_y"]
    elif care_or_youth == "caregiver":
        menarche = row["pds_f5b_p"]
    
    if np.isnan(row["PDS_cat_score"])==False:    
        if menarche == 1.0:
            if row["PDS_cat_score"] == 2.0:
                return "prepubertal"
            if row["PDS_cat_score"] == 3.0:
                return "early pubertal"
            if row["PDS_cat_score"] >= 3.0:
                return "midpubertal"
            
        elif menarche == 4.0:
            if row["PDS_cat_score"] <= 7.0:
                return "late pubertal"
            if row["PDS_cat_score"] == 8.0:
                return "postpubertal"
                     
    elif np.isnan(row["PDS_cat_score"])==True:
        return np.nan

def male_categorical(row: pd.DataFrame) -> pd.DataFrame:
    """
    male: body hair growth + facial hair + voice change
    prepubertal = 3 x
    early pubertal = 4 or 5 (no 3 point response)x
    midpubertal = 6-8 (no point 4 response) x
    late pubertal = 9-11 
    postpubertal = 12 (all 4 point)
    according to Herting et al (2021) Frontiers in Endocrinology
    with minor adjustment to not create cases for which category is not
    well defined (see paper)

    Author: Dominik Kraft
    """
    
    if np.isnan(row["PDS_cat_score"])==False:
        if row["PDS_cat_score"] == 3.0:
            return "prepubertal"
        
        if 4.0 <= row["PDS_cat_score"] <= 5.0:
            return "early pubertal"

        if 6.0 <= row["PDS_cat_score"] <= 8.0:
            return "midpubertal"
   
        if 9.0 <= row["PDS_cat_score"] <= 11.0:
                return "late pubertal"
            
        if row["PDS_cat_score"] == 12.0:
            return "postpubertal"
        
    elif np.isnan(row["PDS_cat_score"])==True:
        return np.nan

def pubertal_scores(care_or_youth: str, demos: pd.DataFrame, path2data: str='../abcd-data-release-5.1/core/', 
                    check_rename: bool=False, drop_nan: bool=True) -> pd.DataFrame:
    """
    This function calculates 'summary' scores derived from the PDS Scale:
    - mean PDS
    -----
    Input:
    - file: ABCD file containing information about perceived puberty without .txt extension. Default: caregiver report
    - demos: dataframe which includes sex information - variable in puberty file has lots of NaN
    -----
    Output:
    - dataframe: merged dataframe containing demographic and puberty information

    adapted from Dominik Kraft
    """

    n_tpts = demos.eventname.nunique()

    # who replied?
    if care_or_youth == "caregiver":
        file = "physical-health/ph_p_pds.csv"
        sex_col = "pubertal_sex_p"

    elif care_or_youth == "youth":
        file = "physical-health/ph_y_pds.csv"
        sex_col = "pds_sex_y"

    df = pd.read_csv(os.path.join(path2data, file), sep=",")
    df = adjust_sub_and_event_column(df)

    if check_rename:
        df.eventname = df.eventname.map(
            {'baselineyear1arm1': 't1', '2yearfollowupyarm1': 't2', '4yearfollowupyarm1': 't3'})

    df = df.merge(demos[["src_subject_id", "sex"]], on="src_subject_id")

    df = df.loc[
        df.eventname.isin(
            ["baselineyear1arm1", "2yearfollowupyarm1", "4yearfollowupyarm1", "t1", "t2", "t3"])]  # choose timepoint
    df.replace(999.0, np.nan, inplace=True)  # replace all "dont know" with np.nan
    df.replace(777.0, np.nan, inplace=True)  # replace all "refuse to answer" with np.nan -- only for youth

    fem = df.loc[df["sex"] == 2.0].copy()  # subset for sex
    men = df.loc[df["sex"] == 1.0].copy()

    # calculate puberty scores based on youth report
    if care_or_youth == "youth":

        substr = "_y"
        # calculate average PDS scores
        fem["PDS_mean" + substr] = fem[["pds_ht2_y", "pds_f4_2_y", "pds_f5_y", "pds_skin2_y", "pds_bdyhair_y"]].mean(
            axis=1,
            skipna=False)
        men["PDS_mean" + substr] = men[["pds_ht2_y", "pds_m4_y", "pds_m5_y", "pds_skin2_y", "pds_bdyhair_y"]].mean(
            axis=1,
            skipna=False)

    # calculate puberty scores based on caregiver report
    elif care_or_youth == "caregiver":

        substr = "_p"
        # calculate average PDS scores
        fem["PDS_mean" + substr] = fem[["pds_1_p", "pds_f4_p", "pds_f5b_p", "pds_3_p", "pds_2_p"]].mean(axis=1,
                                                                                                        skipna=False)
        men["PDS_mean" + substr] = men[["pds_1_p", "pds_m4_p", "pds_m5_p", "pds_3_p", "pds_2_p"]].mean(axis=1,
                                                                                                       skipna=False)

        ## calculate PDS category score
        fem["PDS_cat_score"] = fem[["pds_f4_p", "pds_2_p"]].sum(axis=1, skipna=False)
        men["PDS_cat_score"] = men[["pds_2_p", "pds_m5_p", "pds_m4_p"]].sum(axis=1, skipna=False)

        ## transform PDS category scores to pubertal stages
        fem["PDS_category"] = fem.apply(lambda row: female_categorical(row, care_or_youth="caregiver"), axis=1)
        men["PDS_category"] = men.apply(lambda row: male_categorical(row), axis=1)

    # use only these columns
    pub = pd.concat([fem, men])[["src_subject_id", "eventname", "PDS_mean" + substr, "PDS_cat_score", "PDS_category"]]

    # merge with demos again
    pubdemo = demos.merge(pub, on=["src_subject_id", "eventname"])
    pubdemo = pubdemo.drop_duplicates(subset=["src_subject_id", "eventname"])
    pubdemo = pubdemo.sort_values(['src_subject_id', 'eventname'], ascending=[True, False]).reset_index(drop=True)

    # dropping missing values
    if drop_nan:
        pubdemo = pubdemo.dropna(subset=["PDS_mean" + substr]).reset_index(drop=True)

    # should be only true for longitudinal data
    # if demos.eventname.nunique() == n_tpts:
    #    pubdemo = pubdemo.loc[pubdemo['src_subject_id'].map(pubdemo['src_subject_id'].value_counts()) == n_tpts].reset_index(
    #        drop=True)

    return pubdemo


def reshape_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function reshapes the data into multiple colmuns.
    This makes it a little easier to select specific data to create covariance and response files.

    Parameters
    ----------
    df : pandas.DataFrame
        as input from the ABCD dataset

    Returns
    -------
    dfs : pandas.DataFrame
        a column-wise concatenated dataframe where the columns were renamed to column_tx.
    """

    # rename the timepoint data
    if ('baselineyear1arm1' or '2yearfollowupyarm1' or '4yearfollowupyarm1') in df.eventname.to_list():
        df.eventname = df.eventname.map(
            {'baselineyear1arm1': 'bl', '2yearfollowupyarm1': '2yr', '4yearfollowupyarm1': '4yr'})
    dfs = []
    for i, tpt in enumerate(df.eventname.unique()):
        inter = df[df.eventname == tpt].rename(columns={col: col + f'_{tpt}' for col in df.columns[2:]}).reset_index(
            drop=True)
        dfs.append(inter.iloc[:, 2:] if i > 0 else inter)

    # concat the data

    combined = pd.concat(dfs, axis=1)
    combined.drop(columns="eventname", inplace=True)

    return combined


def make_files(data, x_columns, y_columns, save_dir, site_list=None, split="train"):
    """
    Creates covariate and response files for training normative models

    Parameters
    ----------
    data : pd.DataFrame
    x_columns : list
        Columns that correspond to covariates
    y_columns : list
        Columns that correspond to responses
    save_dir : str/path
        where to save the files
    site_list : list, Default: None
        List with MR-Site identifiers
    split : str, Default: "train"
        Which split to run (train or test).
    """
    cov_save_dir = os.path.join(save_dir, f'cov_files_{split}'); os.makedirs(cov_save_dir, exist_ok=True)
    res_save_dir = os.path.join(save_dir, f'resp_files_{split}'); os.makedirs(res_save_dir, exist_ok=True)
    tpt_id = y_columns[0].split('_')[-1]

    for i, roi in tqdm(enumerate(y_columns)):
        these_coi = x_columns[i] if len(np.array(x_columns).shape) > 1 else x_columns
        cov_data = data[these_coi]

        if (i == 0) & (site_list!=None):
            wo_bspline = 1 + cov_data.shape[-1] + len(site_list)
        elif not site_list: 
            wo_bspline = 1 + cov_data.shape[-1]

        if site_list:
            dm = create_design_matrix(cov_data, intercept=True, site_ids=data['site_bl'], all_sites=site_list, 
                                            xmin=cov_data.iloc[:, 0].min(), xmax=cov_data.iloc[:, 0].max())
        else:
            dm = create_design_matrix(cov_data, xmin=cov_data.iloc[:, 0].min(), xmax=cov_data.iloc[:, 0].max())            

        np.savetxt(os.path.join(cov_save_dir, f'{roi}.txt'), dm[:, :wo_bspline])
        np.savetxt(os.path.join(cov_save_dir, f'{roi}_bspline.txt'), dm)

        resp_data = data[y_columns[i]]
        resp_data.to_csv(os.path.join(res_save_dir, f'{roi}.txt'), header=False, index=False)


def palm_inormal(X, c=None, method='blom', quanti=False):
    """
    Parameters
    ----------
    X : array_like
    c : float, optional
    method : {'blom', 'tukey', 'bliss', 'waerden', 'solar'}, optional
    quanti : bool, optional
    Returns
    -------
    Z : numpy.ndarray

    See also
    --------
    from https://gist.github.com/rmarkello/714e6b80a90229a14edb5d68850f2449
    """

    methods = dict(
        blom=(3 / 8),
        tukey=(1 / 3),
        bliss=(1 / 2),
        waerden=0,
        solar=0
    )
    if method not in methods:
        raise ValueError('Provided method {} invalid. Must be one of {}.'
                         .format(method, methods))
    if c is None:
        c = methods.get(method, 3 / 8)

    if quanti:
        iX = np.argsort(X)
        ri = np.argsort(iX)
        N = len(X)
        p = ((ri - c) / (N - 2 * c + 1))
        Z = np.sqrt(2) * spc.erfinv(2 * p - 1)
    else:
        Z = np.ones_like(X) * np.nan

        for x in range(X.shape[-1]):
            XX = X[:, x]
            ynan = ~np.isnan(XX)
            XX = XX[ynan]

            iX = np.argsort(XX)
            ri = np.argsort(iX)

            N = len(XX)
            p = ((ri + 1 - c) / (N - 2 * c + 1))
            Y = np.sqrt(2) * spc.erfinv(2 * p - 1)

            U, IC = np.unique(Y, return_inverse=True)
            if U.size < N:
                sIC = np.sort(IC)
                dIC = np.diff(np.vstack((sIC, 1)))
                U = np.unique(sIC[dIC == 0])
                for u in range(len(U)):
                    Y[IC == U[u]] = np.mean(Y[IC == U[u]])

            Z[ynan, x] = Y

    return Z


def calibration_descriptives(x):
    """
    See also
    --------
    from
    https://github.com/predictive-clinical-neuroscience/braincharts/blob/master/scripts/nm_utils.py
    """
    n = np.shape(x)[0]
    m1 = np.mean(x)
    m2 = sum((x - m1) ** 2)
    m3 = sum((x - m1) ** 3)
    m4 = sum((x - m1) ** 4)
    s1 = np.std(x)
    skew = n * m3 / (n - 1) / (n - 2) / s1 ** 3
    sdskew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    kurtosis = (n * (n + 1) * m4 - 3 * m2 ** 2 * (n - 1)) / ((n - 1) * (n - 2) * (n - 3) * s1 ** 4)
    sdkurtosis = np.sqrt(4 * (n ** 2 - 1) * sdskew ** 2 / ((n - 3) * (n + 5)))
    semean = np.sqrt(np.var(x) / n)
    sesd = s1 / np.sqrt(2 * (n - 1))
    cd = [skew, sdskew, kurtosis, sdkurtosis, semean, sesd]
    return cd


def get_dummies(sites):
    # function to get the dummies for the scan site.
    # technically, we could use the pd.get_dummies function, but this only works propperly,
    # if all dataframes have the exact same unique sites. If a later, e.g., the 4th year dataframe,
    # contains only a subset of the scan sites, the covariate matrix will not have the same number
    # of columns. That'll lead to a problem. Therefore, we need to create the dummies ourselves.
    # this file should have all
    full = pd.read_csv('ABCD/data/glasser/combined_features.csv')
    dummies = pd.get_dummies(sorted(full.site.unique()), columns=["site"], dtype=float)

    df = pd.DataFrame(columns=dummies.columns)

    for i, s in enumerate(sites):
        df.loc[i, :] = dummies[s].values

    return df

def run_kruskal_wallis(data, cat_data, tpt, cats=['prepubertal', 'early pubertal', 'midpubertal', 'late pubertal', 'postpubertal']):
    vals = [np.array(data)[cat_data[cat_data[f'PDS_category_{tpt}']==cat].index] for cat in cats]
    non_empty_idx = [int(i) for i,x in enumerate(vals) if len(x) > 0]
    res = kruskal(*[vals[i] for i in non_empty_idx])
    return vals, res, non_empty_idx

def make_anova_plots(data, metric, figsize=(20, 8), sharey=True):
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=sharey)

    if metric == 'MSLL':
        data = data[data[metric] < 1000]

    if metric == 'kurtosis':
        data = data[data[metric] < 20]
    
    for s, sex in enumerate(data.sex.unique()):
        sex_data = data[data.sex == sex]    
        # main effect model
        sns.stripplot(data=sex_data, x='model', y=metric, hue='model', ax=axes[s, 0], legend=False, alpha=.1, palette='gray')
        sns.pointplot(data=sex_data, x='model', y=metric, ax=axes[s, 0], zorder=10)
        axes[s, 0].set_title(f"model = {sex}")
    
        # main effect timepoint
        sns.stripplot(data=sex_data, x='timepoint', y=metric, hue='model', ax=axes[s, 1], legend=False, alpha=.1, palette='gray')
        sns.pointplot(data=sex_data, x='timepoint', y=metric, ax=axes[s, 1], zorder=10)
        axes[s, 1].set_title(f"timepoint")
    
        # interaction: model*timepoint
        sns.stripplot(data=sex_data, x='model', y=metric, hue='timepoint', ax=axes[s, 2], dodge=True, legend=False, alpha=.1, palette='gray')
        sns.pointplot(data=sex_data, x='model', y=metric, hue='timepoint', ax=axes[s, 2], dodge=.4, zorder=10)
        axes[s, 2].set_title(f"model({sex})*timepoint")

    return fig, axes


def make_anova_plots_full(data, metric, figsize=(20, 8), sharey='row'):
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharey=sharey)

    axes.flatten()[-1].remove()

    overall_mean = data[metric].mean()
    # main effect model
    sns.pointplot(data=data, x='model', y=metric, ax=axes[0, 0])
    axes[0, 0].set_title(f"model")

    # main effect sex
    sns.pointplot(data=data, x='sex', y=metric, ax=axes[0, 1])
    axes[0, 1].set_title(f"sex")

    # main effect timepoint
    sns.pointplot(data=data, x='timepoint', y=metric, ax=axes[0, 2])
    axes[0, 2].set_title(f"timepoint")

    # interaction: model*sex
    sns.pointplot(data=data, x='model', y=metric, hue='sex', ax=axes[1, 0])
    axes[1, 0].set_title(f"model*sex")

    # interaction: model*timepoint
    sns.pointplot(data=data, x='model', y=metric, hue='timepoint', ax=axes[1, 1])
    axes[1, 1].set_title(f"model*timepoint")

    # interaction: sex*timepoint
    sns.pointplot(data=data, x='sex', y=metric, hue='timepoint', ax=axes[1, 2])
    axes[1, 2].set_title(f"sex*timepoint")

    # 3-way interactions
    # 1. AgeSexModel
    model_ids = data.model.unique()
    sns.pointplot(data=data[data.model == model_ids[0]], x='timepoint', y=metric, hue='sex', ax=axes[2, 0])
    axes[2, 0].set_title(f"model = {model_ids[0]}")
    # 2. ThicknessAgeSexModel
    sns.pointplot(data=data[data.model == model_ids[1]], x='timepoint', y=metric, hue='sex', ax=axes[2, 1])
    axes[2, 1].set_title(f"model = {model_ids[1]}")

    return fig, axes
