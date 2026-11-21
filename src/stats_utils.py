import numpy as np
from scipy import stats


def kruskal_wallis(groups_list):
    return stats.kruskal(*groups_list)


def safe_alexander_govern(groups_list):
    """Heteroscedastic omnibus ANOVA; jitters zero-variance groups to avoid division by zero."""
    safe_g = [g + (np.random.normal(0, 1e-9, len(g)) if np.std(g) < 1e-12 else 0.0) for g in groups_list]
    res = stats.alexandergovern(*safe_g)
    return res.statistic, res.pvalue


def run_posthoc_welch(base_arr, comp_dict):
    raw_p, rows = [], []
    for name, other in comp_dict.items():
        if np.std(base_arr) < 1e-12 and np.std(other) < 1e-12:
            t_s, p_v = 0.0, 1.0
        else:
            t_s, p_v = stats.ttest_ind(base_arr, other, equal_var=False)
        raw_p.append(p_v)
        rows.append((name, base_arr.mean() - other.mean(), t_s, p_v))
    order = np.argsort(raw_p)
    adj_p = np.zeros(len(raw_p))
    for rank, idx in enumerate(order):
        adj_p[idx] = min(1.0, raw_p[idx] * (len(raw_p) - rank))
    return rows, adj_p
