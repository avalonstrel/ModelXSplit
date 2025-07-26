import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Lasso
from scipy.optimize import minimize

class LightGBMModel:
    def __init__(self, model_type, X, y, D, M, M_tilde):
        self.model_type = model_type
        self.X, self.y = X, y
        self.D = D
        self.M, self.M_tilde = M, M_tilde
        self.X_prime = self.X - self.M @ self.D
        self.beta = None
        self.gamma = None
        self.gamma_tilde = None
        self.lambda_ = 1.0
        self.nu_inv_ = 1.0
        assert model_type in ["logistic", "linear"], f"Model type:{model_type} not supported."

    def train_lightgbm(self, X_combined, y):
        """Train a LightGBM model for prediction loss."""
        train_data = lgb.Dataset(X_combined, label=y)
        params = {'objective': 'binary' if self.model_type == "logistic" else 'regression',
                  'learning_rate': 0.05,
                  'num_leaves': 31}
        model = lgb.train(params, train_data, num_boost_round=100)
        return model

    def regularize_parameters(self, beta, gamma, gamma_tilde):
        """Solve for beta, gamma, gamma_tilde with L1/L2 regularization."""
        def objective(params):
            beta, gamma, gamma_tilde = np.split(params, [self.D.shape[1], self.D.shape[1] + self.M.shape[1]])
            l2_penalty = self.nu_inv_ * (
                np.linalg.norm(self.D @ beta - gamma, ord=2)**2 + 
                np.linalg.norm(self.D @ beta - gamma_tilde, ord=2)**2
            )
            l1_penalty = self.lambda_ * (np.linalg.norm(gamma, ord=1) + np.linalg.norm(gamma_tilde, ord=1))
            return l2_penalty + l1_penalty

        # Initial guess
        init_params = np.hstack([beta, gamma, gamma_tilde])

        # Solve optimization problem
        result = minimize(objective, init_params, method="L-BFGS-B")
        beta, gamma, gamma_tilde = np.split(result.x, [self.D.shape[1], self.D.shape[1] + self.M.shape[1]])
        return beta, gamma, gamma_tilde

    def solve(self, lambda_val, nu_inv_val, max_iterations=100):
        """Iterative solver using LightGBM and optimization for regularization."""
        self.lambda_ = lambda_val
        self.nu_inv_ = nu_inv_val

        # Initialize parameters
        beta = np.random.randn(self.D.shape[1])
        gamma = np.random.randn(self.M.shape[1])
        gamma_tilde = np.random.randn(self.M_tilde.shape[1])

        for iteration in range(max_iterations):
            # Step 1: Train LightGBM with current parameters
            X_combined = self.X_prime @ beta + self.M @ gamma + self.M_tilde @ gamma_tilde

            
            model = self.train_lightgbm(X_combined, self.y)

            # Step 2: Update regularization parameters
            beta, gamma, gamma_tilde = self.regularize_parameters(beta, gamma, gamma_tilde)

            # Evaluate loss
            loss = self.cv_evaluate_loss(beta, gamma, gamma_tilde)
            print(f"Iteration {iteration + 1}, Loss: {loss:.4f}")

        return beta, gamma, gamma_tilde
    
    def cv_evaluate_loss(
        self, beta, gamma, gamma_tilde, X, y, D, M, M_tilde, lambda_val, nu_inv_val
    ):
        """Evaluate combined loss (prediction + regularization)."""
        if beta is None or gamma is None or gamma_tilde is None:
            return np.inf

        # Compute X_prime
        X_prime = X - M @ D

        # Compute prediction
        X_pred = X_prime @ beta + M @ gamma + M_tilde @ gamma_tilde

        # Compute the prediction loss
        if self.model_type == "logistic":
            # Logistic loss
            loss_val = -np.sum(
                y * np.log1p(np.exp(-X_pred)) + (1 - y) * np.log1p(np.exp(X_pred))
            )
        else:
            # MSE loss
            loss_val = np.linalg.norm(X_pred - y, ord=2) ** 2 / len(X)

        # Compute L2 regularization
        l2_reg_val = nu_inv_val * (
            np.linalg.norm(D @ beta - gamma, ord=2) ** 2
            + np.linalg.norm(D @ beta - gamma_tilde, ord=2) ** 2
        )

        # Compute L1 regularization
        l1_reg_val = lambda_val * (
            np.linalg.norm(gamma, ord=1) + np.linalg.norm(gamma_tilde, ord=1)
        )

        # Combine all loss components
        return loss_val + l2_reg_val + l1_reg_val