from numpy import cov
import scipy.stats as stats


def covariance(d1, d2, verbose=False):
    result = cov(d1, d2)
    if verbose:
        print('Covariance:', result)
    return result


def get_correlation(d1, d2, spearman=False, kendall=False, verbose=False):
    result, _ = stats.spearmanr(d1, d2) if spearman else stats.kendalltau(d1, d1) if kendall else stats.pearsonr(d1, d2)
    if verbose:
        print('Result:', result,
              '.  There is a ',
              'Strong' if abs(result) > 0.5 else 'Weak',
              'Positive' if result > 0 else 'Negative',
              'Correlation')
    return result

