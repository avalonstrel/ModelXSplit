from splitfdr.utils.misc import normalize_func, select, random_split
from splitfdr.construct import construct_spllitmodelx_matrix
from splitfdr.W_statistics.modelx_statistics import compute_W_stat


class SplitModelXFilter:
    """
    Performs ModelX-based inference, from start to finish.

    Parameters
    """

    def __init__(
        self,
    ):
        """
        Initialize the class.
        """

        # self.q = q
        # self.method = method
        pass

    def forward(
        self,
        X,
        y,
        D,
        Sigma,
        Sigma_M,
        split_ratio,
        con_type,
        model_type,
        lambdas,
        nus,
        q=0.10,
        Z_types=("LCD"),
        W_types=("st"),
        T_types=("knockoff"),
        normalize=True,
        norm_type="l2",
        is_modelX=False,
        aug_n=0
    ):
        """
        Runs the modelx filter; returns whether each feature was rejected.

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
        Z_types : str
            The type of Z statistics used in the method. Now support ['db', 'cf', 'sep']
        W_types : str
            The type of W statistics used in the method. Now support ['bc', 'lcd', 'lsm']
        T_types: str
            The type to decide the Threshold.
        is_ModelX: bool
            Whether to treat the model as original ModelX which just ignore the D.

        """
        # normalize the data
        if normalize:
            X, y = normalize_func(X), normalize_func(y)
        n, m, p = X.shape[0], D.shape[0], X.shape[1]

        # Get a solved beta first which will used in the W statistics computation
        # TODO Fake split now
        X1, y1, X2, y2 = random_split(X, y, split_ratio=split_ratio)
        # X1, y1 = X, y
        # X2, y2 = X, y
        n1 = len(X1)

        # Generate M and M_tilde
        alpha, M, M_tilde = construct_spllitmodelx_matrix(
            X, D, Sigma, Sigma_M, is_modelX=is_modelX, con_type=con_type, aug_n=aug_n
        )
        
        # Compute the W statistics
        Ws, cv_lambda_vals, cv_nu_vals = compute_W_stat(
            model_type, Z_types, W_types, X, y, D, M, M_tilde, lambdas, nus
        )

        # Compute the chosen S and results
        results = {}

        for Z_type in Z_types:
            for W_type in W_types:
                for T_type in T_types:
                    type_key = f"{Z_type}_{W_type}"
                    S = select(Ws[type_key], q, T_type)
                    # print(f'W:{W.shape}, S:{S.shape}')
                    results[f"lambda_{type_key}"] = cv_lambda_vals[type_key]
                    results[f"nu_{type_key}"] = cv_nu_vals[type_key]
                    results[f"W_{type_key}"] = Ws[type_key]
                    results[f"S_{type_key}_{T_type}"] = S
        return results
