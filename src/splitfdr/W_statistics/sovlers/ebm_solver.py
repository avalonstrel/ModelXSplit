import numpy as np
from interpret.glassbox import CExplainableBoostingRegressor, CExplainableBoostingClassifier

class EBMModel:
    def __init__(self, model_type, X, y, D, M, M_tilde):
        self.model_type = model_type
        self.X = np.array(X, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64).reshape(-1, 1)
        self.D = np.array(D, dtype=np.float64)
        self.M = np.array(M, dtype=np.float64)
        self.M_tilde = np.array(M_tilde, dtype=np.float64)
        self.X_prime = self.X - self.M @ self.D

        # Prepare combined input features
        self.combined_input = np.hstack([
            self.X_prime,
            self.M,
            self.M_tilde
        ])
        beta_num, M_num = self.X_prime.shape[1], self.M.shape[1]
        self.beta_indices = np.arange(beta_num)
        self.gamma_indices = np.arange(beta_num, beta_num + M_num)
        self.gamma_tilde_indices = np.arange(beta_num + M_num, beta_num + 2 * M_num)
        # Initialize EBM model
        self.create_model(1e-2)
        self.feature_importances_ = None

    def create_model(self, lambda_val):
        if self.model_type == "logistic":
            self.ebm_model = CExplainableBoostingClassifier(
                random_state=42,
                max_rounds=500, 
                beta_indices=self.beta_indices,
                gamma_indices=self.gamma_indices,
                gamma_tilde_indices=self.gamma_tilde_indices,
                D=self.D,  # Example D matrix
                lambda_reg=lambda_val,
                interactions=0)
        else:
            self.ebm_model = CExplainableBoostingRegressor(
                random_state=42, 
                max_rounds=500,
                beta_indices=self.beta_indices,
                gamma_indices=self.gamma_indices,
                gamma_tilde_indices=self.gamma_tilde_indices,
                D=self.D,  # Example D matrix
                lambda_reg=lambda_val,
                interactions=0)

    def train_model(self, lambda_val=1e-2, nu_inv_val=1e-2,):
        self.create_model(lambda_val)
        # EBM expects 1d y for classification/regression
        y_train = self.y.ravel()
        print("Check", lambda_val)
        self.ebm_model.lambda_reg = lambda_val
        self.ebm_model.fit(self.combined_input, y_train)
        # self.feature_importances_ = self.ebm_model.feature_importances_

    def solve(self, lambda_val=None, nu_inv_val=None, epochs=1, lr=None):
        # Keep signature for compatibility; lambda/nu_inv/lr unused
        self.train_model(lambda_val, nu_inv_val)

        feature_scores = np.array(self.ebm_model.term_scores_)
        # feature_scores = self.ebm_model.term_importances()
        # print("Check", len(feature_scores))
        n_beta = self.X_prime.shape[1]
        n_gamma = self.M.shape[1]
        n_gamma_tilde = self.M_tilde.shape[1]

        beta = feature_scores[:n_beta].mean(-1)
        gamma = feature_scores[n_beta:n_beta+n_gamma].mean(-1)
        gamma_tilde = feature_scores[n_beta+n_gamma:n_beta+n_gamma+n_gamma_tilde].mean(-1)
        # print("Check 1",beta.shape, gamma.shape, gamma_tilde.shape)

        return beta, gamma, gamma_tilde

    def cv_evaluate_loss(
        self, beta_imp, gamma_imp, gamma_tilde_imp, X, y, D, M, M_tilde, lambda_val=None, nu_inv_val=None
    ):
        return 0
        # Construct combined input for test
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        D = np.array(D, dtype=np.float32)
        M = np.array(M, dtype=np.float32)
        M_tilde = np.array(M_tilde, dtype=np.float32)
        X_prime = X - M @ D

        combined_input = np.hstack([X_prime, M, M_tilde])
        preds = self.ebm_model.predict(combined_input)
        if self.model_type == "logistic":
            from sklearn.metrics import log_loss
            loss = log_loss(y.ravel(), preds)
        else:
            from sklearn.metrics import mean_squared_error
            loss = mean_squared_error(y.ravel(), preds)
        return loss

    def get_feature_importances(self):
        if self.feature_importances_ is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        n_beta = self.X_prime.shape[1]
        n_gamma = self.M.shape[1]
        n_gamma_tilde = self.M_tilde.shape[1]
        importances = self.feature_importances_
        beta_importance = importances[:n_beta]
        gamma_importance = importances[n_beta:n_beta+n_gamma]
        gamma_tilde_importance = importances[n_beta+n_gamma:]
        return beta_importance, gamma_importance, gamma_tilde_importance