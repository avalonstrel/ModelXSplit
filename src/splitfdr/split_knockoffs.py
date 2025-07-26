import warnings
import numpy as np

from splitfdr.utils.misc import normalize_func, select, random_split
from splitfdr.intercept import get_intercept
from splitfdr.W_statistics.splitknockoffs_statistics import compute_W_solpath_statistics

class SplitKnockoffsFilter:
    """
    Performs knockoff-based inference, from start to finish.

    This wraps both the ``knockoffs.KnockoffSampler`` and 
    ``knockoff_stats.FeatureStatistic`` classes.

    Parameters 
    """

    def __init__(
        self, 
    ):
        """
        Initialize the class.
        """
        # @TODO Just left as an extention interface.
        

    def forward(
        self,
        X,
        y,
        D,
        split_ratio,
        intercept_method,
        lambdas,
        q=0.10,
        methods=('knockoff'),
        W_types=('st'),
        normalize=True,
    ):
        """
        Runs the knockoff filter; returns whether each feature was rejected.
        
        Parameters
        ----------
        X : np.ndarray
            ``(n, p)``-shaped design matrix
        y : np.ndarray
            ``(n,)``-shaped response vector
        D : np.ndarrays
            ``(m, p)``-shaped transformation matrix
        split_ratio: float
            The split ratio of the dataset
        fdr : float
            The desired level of false discovery rate control.
        fstat_kwargs : dict
            Extra kwargs to pass to the feature statistic ``fit`` function,
            excluding the required arguments.
        knockoff_kwargs : dict
            Extra kwargs for instantiating the knockoff sampler argument if
            the ksampler argument is a string identifier. This can be
            the empty dict for some identifiers such as "gaussian" or "fx",
            but additional keyword arguments are required for complex samplers
            such as the "metro" identifier. Defaults to {}
        """
        # update the knockoff kwargs
        if normalize:
            X, y = normalize_func(X), normalize_func(y)
        n, m, p = X.shape[0], D.shape[0], X.shape[1]
        
        # Split X,y to X1, y1, X2, y2
        X1, y1, X2, y2 = random_split(X, y, split_ratio)

        # get the beta interception on D_1
        # shape[len(lambdas), p]
        # @TODO
        # Only implement the cv_all and cv_screen
        nus = np.power(10, np.arange(0, 2, 0.2))
        if intercept_method in ['cv_all']:
            nus = [0]
            X1, y1, X2, y2 = random_split(X, y, split_ratio)
            n1 = len(X1)
            # print(f'qnu val: {nu}')
            beta_intercepts, stats = get_intercept(X1, y1, D, 0, None, 
                                            method=intercept_method, n2=n-n1)
        elif intercept_method in ['path']:
            beta_intercepts, stats = get_intercept(X1, y1, D, 0, None, 
                                            method='cv_all', n2=n-n1)
            if 'beta_supp' in stats:
                X = X[:, stats['beta_supp']]
                D = D[stats['gamma_supp'], stats['beta_supp']]
        
        # Compute the Split Knockoffs Statistics
        if 'nu' in stats:
            nu = stats['nu']

        X_new, D_new = X2, D
        if 'beta_supp' in stats:
            X_new = X2[:, stats['beta_supp']]
            D_new = D[stats['gamma_supp'], stats['beta_supp']]
        print(f'compute statistics nu val: {nu}')
        
        # Compute W statistics
        Ws = compute_W_solpath_statistics(X_new, y2, D_new, 
                        beta_intercepts, nu, lambdas, 
                        stats, W_types)
        results = {}
        for W_type in W_types:
            for method in methods:
                S = select(Ws[W_type], q, method)
                results[f'W_{W_type}'] = Ws[W_type]
                results[f'S_{W_type}_{method}'] = S
        return results
