import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier

class LGBModel:
    def __init__(self, model_type, X, y, D, M, M_tilde):
        self.model_type = model_type
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32).reshape(-1, 1)
        self.D = np.array(D, dtype=np.float32)
        self.M = np.array(M, dtype=np.float32)
        self.M_tilde = np.array(M_tilde, dtype=np.float32)
        self.X_prime = self.X - self.M @ self.D

        # Prepare combined input features (as in original code)
        # No beta/gamma parameters: just stack the raw features
        self.combined_input = np.hstack([
            self.X_prime,         # Features corresponding to 'beta'
            self.M,               # Features corresponding to 'gamma'
            self.M_tilde          # Features corresponding to 'gamma_tilde'
        ])

        # Initialize LGBM model
        self.create_model(1e-2, 1e-2)
        self.feature_importances_ = None

    def create_model(self, lambda_val, nu_inv_val):
        if self.model_type == "logistic":
            self.lgb_model = LGBMClassifier(
                random_state=42,
                tree_learner="serial",
                custom_reg_lambda=lambda_val,
                custom_reg_nu_inv=nu_inv_val,)
        else:
            self.lgb_model = LGBMRegressor(
                D_type="D_1",
                D_rows=self.D.shape[0],
                D_cols=self.D.shape[1],
                tree_learner="serial",
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=1,
                num_leaves=2,
                max_bin=64,
                bagging_freq=0,
                feature_fraction=1.0,             # use all features for every tree
                feature_fraction_bynode=1.0,      # use all features for every split
                feature_fraction_bylevel=1.0,
                reg_alpha=1, reg_lambda=1,
                custom_reg_lambda=lambda_val,
                custom_reg_nu_inv=nu_inv_val,
                random_state=27, verbose=-1
            )
    def train_model(self, lambda_val, nu_inv_val):
        self.create_model(lambda_val, nu_inv_val)
        # Train the LGBM model on the combined input

        self.lgb_model.fit(self.combined_input, self.y.ravel())
        # train_loss = self.cv_evaluate_loss(None, None, None, self.X, self.y, self.D, self.M, self.M_tilde,)
        # print("Train Loss: ", train_loss)
        # self.feature_importances_ = self.lgb_model.feature_importances_
        # self.feature_importances_ = self.lgb_model.dir_importances_
        self.feature_importances_ = self.lgb_model.booster_.feature_importance(importance_type='gain')
        # self.feature_importances_ = self.lgb_model.booster_.directional_importance(importance_type='gain')
        
        # shap_values = self.lgb_model.predict(self.combined_input, pred_contrib=True)
        # shap_values = np.array(shap_values)[:, :-1]
        # self.feature_importances_ = np.abs(shap_values).mean(axis=0)


    def solve(self, lambda_val=None, nu_inv_val=None, epochs=1, lr=None):
        # Keep signature for compatibility; lambda/nu_inv/lr unused
        self.train_model(lambda_val, nu_inv_val)
        # Map feature importances to original groups
        n_beta = self.X_prime.shape[1]
        n_gamma = self.M.shape[1]
        n_gamma_tilde = self.M_tilde.shape[1]
        importances = self.feature_importances_

        beta_importance = importances[:n_beta]
        gamma_importance = importances[n_beta:n_beta+n_gamma]
        gamma_tilde_importance = importances[n_beta+n_gamma:]
        # print("11", gamma_importance, gamma_tilde_importance)
        # sss
        return beta_importance, gamma_importance, gamma_tilde_importance

    def cv_evaluate_loss(
        self, beta_imp, gamma_imp, gamma_tilde_imp, X, y, D, M, M_tilde, lambda_val=None, nu_inv_val=None
    ):
        # For evaluation: use the same combined input construction
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        D = np.array(D, dtype=np.float32)
        M = np.array(M, dtype=np.float32)
        M_tilde = np.array(M_tilde, dtype=np.float32)
        X_prime = X - M @ D

        combined_input = np.hstack([X_prime, M, M_tilde])
        preds = self.lgb_model.predict(combined_input)
        if self.model_type == "logistic":
            from sklearn.metrics import log_loss
            loss = log_loss(y.ravel(), preds)
        else:
            from sklearn.metrics import mean_squared_error
            loss = mean_squared_error(y.ravel(), preds)
        print("Test Loss: ", loss)
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