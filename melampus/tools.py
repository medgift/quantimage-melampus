import numpy as np
import pandas as pd

def find_correlated_features(df, cut=0.9, method='pearson'):
    """
    Checks if the correlation coefficient between a pair features exceeds the threshold value 'cut'.
    For such a highly correlated pair of features, the feature with the higher mean value is considered for dropping.
    The final drop decision algorithm ensures that no feature is dropped accidentally although it would be the uniquely
    remaining feautre in a highly correlated pair.
    Based on  https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6
    :param df: pandas dataframe with feature values
    :param cut: correlation cut-off
    :param method: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
    :return: list of variable names to be dropped
    """
    corr_mtx = df.corr(method=method).abs()
    avg_corr = corr_mtx.mean(axis=1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(bool))

    dropcols = list()

    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target',
                                 'v2.target', 'corr', 'drop']))

    for row in range(len(up) - 1):
        col_idx = row + 1
        for col in range(col_idx, len(up)):
            if (corr_mtx.iloc[row, col] > cut):
                if (avg_corr.iloc[row] > avg_corr.iloc[col]):
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else:
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]

                s = pd.Series([corr_mtx.index[row],
                               up.columns[col],
                               avg_corr[row],
                               avg_corr[col],
                               up.iloc[row, col],
                               drop],
                              index=res.columns)

                res = res.append(s, ignore_index=True)

    dropcols_names = calcDrop(res)

    return (dropcols_names)


def calcDrop(res):
    """
    Helper function for find_correlated_features.
    :param res: results dataframe from 'find_correlated_features'
    :return: list of variable names to be dropped
    """
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))

    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))

    # Drop any variables in same row as a keep variable
    p = res[res['v1'].isin(keep) | res['v2'].isin(keep)][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop
    poss_drop = list(set(poss_drop).difference(set(drop)))

    # subset res dataframe to include possible drop pairs
    m = res[res['v1'].isin(poss_drop) | res['v2'].isin(poss_drop)][['v1', 'v2', 'drop']]

    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)

    return drop
