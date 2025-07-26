import cvxpy as cp
import numpy as np

class ModelXCVXSolver:
    def __init__(self, model_type, X, y, D, M, M_tilde) -> None:
        n, m = M.shape
        p = X.shape[1]
        self.model_type = model_type
        self.X, self.y = X, y
        self.D = D
        self.M, self.M_tilde = M, M_tilde
        self.X_prime = self.X - self.M @ self.D

        # print("X prime", self.X_prime.sum(), (M-M_tilde).sum())
        # Gammas
        self.beta = cp.Variable(p)
        self.gamma = cp.Variable(m)
        self.gamma_tilde = cp.Variable(m)
        self.lambda_val = cp.Parameter(nonneg=True)
        self.nu_inv_val = cp.Parameter(nonneg=True)
        self.model_type = model_type
        assert model_type in [
            "logistic",
            "linear",
        ], f"Model type:{model_type} not supported."
        if model_type == "logistic":
            # model_func = torch.sigmoid
            self.loss_func = self.logistic_loss
        else:
            self.loss_func = self.mse_loss
        self.prob = cp.Problem(
            cp.Minimize(
                self.__evaluate_loss(
                    self.X,
                    self.X_prime,
                    self.y,
                    self.D,
                    self.M,
                    self.M_tilde,
                    self.lambda_val,
                    self.nu_inv_val,
                )
            )
        )

    def logistic_loss(self, X, X_prime, y, M, M_tilde):
        X_pred = X_prime @ self.beta + M @ self.gamma
        X_pred_tilde = X_prime @ self.beta + M_tilde @ self.gamma_tilde
        log_likelihood = cp.sum(
            cp.multiply(y, X_pred)
            - cp.logistic(X_pred)
            + cp.multiply(y, X_pred_tilde)
            - cp.logistic(X_pred_tilde)
        )
        return (-log_likelihood) / (len(X_prime)*2) 
        # X_pred = X_prime @ self.beta + M @ self.gamma + M_tilde @ self.gamma_tilde
        # # X_pred_tilde = X_prime @ self.beta + M_tilde @ self.gamma_tilde
        # log_likelihood = cp.sum(
        #     cp.multiply(y, X_pred)
        #     - cp.logistic(X_pred)
        # )
        # X_beta_pred = X @ self.beta
        # beta_log_likelihood = cp.sum(
        #     cp.multiply(y, X_beta_pred)
        #     - cp.logistic(X_beta_pred)
        # )
        # return (- log_likelihood ) / (len(X_prime)) 

    def mse_loss(self, X, X_prime, y, M, M_tilde):
        X_pred = X_prime @ self.beta + M @ self.gamma + M_tilde @ self.gamma_tilde
        return cp.norm2(X_pred - y) ** 2 / len(X_prime)

    def __evaluate_loss(self, X, X_prime, y, D, M, M_tilde, lambda_val, nu_inv_val):
        loss_val = self.loss_func(X, X_prime, y, M, M_tilde)
        l2_reg_val = nu_inv_val * (
            cp.norm2(D @ self.beta - self.gamma) ** 2
            + cp.norm2(D @ self.beta - self.gamma_tilde) ** 2
        )
        l1_reg_val = lambda_val * (cp.norm1(self.gamma) + cp.norm1(self.gamma_tilde))
        return loss_val + l2_reg_val + l1_reg_val

    def cv_logistic_loss(self, beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde):
        def np_logistic(x):
            return np.log1p(np.exp(x))

        X_pred = X_prime @ beta + M @ gamma
        X_pred_tilde = X_prime @ beta + M_tilde @ gamma_tilde
        log_likelihood = np.sum(
            np.multiply(y, X_pred)
            - np_logistic(X_pred)
            + np.multiply(y, X_pred_tilde)
            - np_logistic(X_pred_tilde)
        )
        return -log_likelihood / (len(X_prime) * 2)
        # X_pred = X_prime @ beta + M @ gamma + M_tilde @ gamma_tilde
        # log_likelihood = np.sum(
        #     np.multiply(y, X_pred)
        #     - np_logistic(X_pred)
        # )
        # X_beta_pred = X @ beta
        # beta_log_likelihood = np.sum(
        #     np.multiply(y, X_beta_pred)
        #     - np_logistic(X_beta_pred)
        # )
        # return -(log_likelihood) / (len(X_prime))

    def cv_mse_loss(self, beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde):
        X_pred = X_prime @ beta + M @ gamma + M_tilde @ gamma_tilde
        return np.linalg.norm(X_pred - y, ord=2) ** 2 / len(X_prime)

    def cv_evaluate_loss(
        self, beta, gamma, gamma_tilde, X, y, D, M, M_tilde, lambda_val, nu_inv_val
    ):
        if beta is None or gamma is None or gamma_tilde is None:
            return np.inf
        X_prime = X - M @ D
        if self.model_type == "logistic":
            cv_loss_func = self.cv_logistic_loss
        else:
            cv_loss_func = self.cv_mse_loss
        loss_val = cv_loss_func(beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde)
        l2_reg_val = nu_inv_val * (
            np.linalg.norm(D @ beta - gamma, ord=2) ** 2
            + np.linalg.norm(D @ beta - gamma_tilde, ord=2) ** 2
        )
        l1_reg_val = lambda_val * (
            np.linalg.norm(gamma, ord=1) + np.linalg.norm(gamma_tilde, ord=1)
        )
        # print("CV losses", nu_inv_val, lambda_val, loss_val, l2_reg_val, l1_reg_val)
        return loss_val + l2_reg_val + l1_reg_val

    # Adam n_iters=100, lr=1
    def solve(self, lambda_val, nu_inv_val):
        self.lambda_val.value = lambda_val
        self.nu_inv_val.value = nu_inv_val
        
        self.prob.solve(solver=cp.MOSEK, verbose=False)
        
        # self.prob.solve(verbose=False)
        # print("Prob:", self.prob.value)
        # print("Solved Results", self.beta.value, self.gamma.value, self.gamma_tilde.value)
        return self.beta.value, self.gamma.value, self.gamma_tilde.value

# Combine features
class ModelXCVXCFSolver:
    def __init__(self, model_type, X, y, D, M, M_tilde) -> None:
        n, m = M.shape
        p = X.shape[1]
        self.model_type = model_type
        self.X, self.y = X, y
        self.D = D
        self.M, self.M_tilde = M, M_tilde
        self.X_prime = self.X - self.M @ self.D
        # Gammas
        self.beta = cp.Variable(p)
        self.gamma = cp.Variable(m)
        self.gamma_tilde = cp.Variable(m)
        self.lambda_val = cp.Parameter(nonneg=True)
        self.nu_inv_val = cp.Parameter(nonneg=True)
        self.model_type = model_type
        assert model_type in [
            "logistic",
            "linear",
        ], f"Model type:{model_type} not supported."
        if model_type == "logistic":
            self.loss_func = self.logistic_loss
        else:
            self.loss_func = self.mse_loss
        self.prob = cp.Problem(
            cp.Minimize(
                self.__evaluate_loss(
                    self.X,
                    self.X_prime,
                    self.y,
                    self.D,
                    self.M,
                    self.M_tilde,
                    self.lambda_val,
                    self.nu_inv_val,
                )
            )
        )

    def logistic_loss(self, X, X_prime, y, M, M_tilde):
        X_pred = X_prime @ self.beta + M @ self.gamma + M_tilde @ self.gamma_tilde
        log_likelihood = cp.sum(
            cp.multiply(y, X_pred)
            - cp.logistic(X_pred)
        )
        return (-log_likelihood) / (len(X_prime)) 

    def mse_loss(self, X, X_prime, y, M, M_tilde):
        X_pred = X_prime @ self.beta + M @ self.gamma + M_tilde @ self.gamma_tilde
        return cp.norm2(X_pred - y) ** 2 / len(X_prime)

    def __evaluate_loss(self, X,  X_prime, y, D, M, M_tilde, lambda_val, nu_inv_val):
        loss_val = self.loss_func(X, X_prime, y, M, M_tilde)
        l2_reg_val = nu_inv_val * (
            cp.norm2(D @ self.beta - self.gamma - self.gamma_tilde) ** 2
        )
        l1_reg_val = lambda_val * (cp.norm1(self.gamma) + cp.norm1(self.gamma_tilde))
        return loss_val + l2_reg_val + l1_reg_val

    def cv_logistic_loss(self, beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde):
        def np_logistic(x):
            return np.log1p(np.exp(x))

        X_pred = X_prime @ beta + M @ gamma + M_tilde @ gamma_tilde
        log_likelihood = np.sum(
            np.multiply(y, X_pred)
            - np_logistic(X_pred)
        )
        return -log_likelihood / (len(X_prime))
        
    def cv_mse_loss(self, beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde):
        X_pred = X_prime @ beta + M @ gamma + M_tilde @ gamma_tilde
        return np.linalg.norm(X_pred - y, ord=2) ** 2 / len(X_prime)

    def cv_evaluate_loss(
        self, beta, gamma, gamma_tilde, X, y, D, M, M_tilde, lambda_val, nu_inv_val
    ):
        if beta is None or gamma is None or gamma_tilde is None:
            return np.inf
        X_prime = X - M @ D
        if self.model_type == "logistic":
            cv_loss_func = self.cv_logistic_loss
        else:
            cv_loss_func = self.cv_mse_loss
        loss_val = cv_loss_func(beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde)
        l2_reg_val = nu_inv_val * (
            np.linalg.norm(D @ beta - gamma - gamma_tilde, ord=2) ** 2
        )
        l1_reg_val = lambda_val * (
            np.linalg.norm(gamma, ord=1) + np.linalg.norm(gamma_tilde, ord=1)
        )
        # print("CV losses", lambda_val, nu_inv_val, loss_val, l2_reg_val, l1_reg_val)
        return loss_val + l2_reg_val + l1_reg_val

    # Adam n_iters=100, lr=1
    def solve(self, lambda_val, nu_inv_val):
        self.lambda_val.value = lambda_val
        self.nu_inv_val.value = nu_inv_val
        self.prob.solve(solver=cp.MOSEK, verbose=False)
        # print("Prob:", self.prob.value)
        # print("Solved Results", self.beta.value, self.gamma.value, self.gamma_tilde.value)
        return self.beta.value, self.gamma.value, self.gamma_tilde.value


# Seperate solve
class ModelXCVXSepSolver:
    def __init__(self, model_type, X, y, D, M, M_tilde) -> None:
        n, m = M.shape
        p = X.shape[1]
        self.model_type = model_type
        self.X, self.y = X, y
        self.D = D
        self.M, self.M_tilde = M, M_tilde
        self.X_prime = self.X - self.M @ self.D
        
        # Gammas
        self.beta = cp.Variable(p)
        self.gamma = cp.Variable(m)
        self.gamma_tilde = cp.Variable(m)
        self.lambda_val = cp.Parameter(nonneg=True)
        self.nu_inv_val = cp.Parameter(nonneg=True)
        self.model_type = model_type
        assert model_type in [
            "logistic",
            "linear",
        ], f"Model type:{model_type} not supported."
        if model_type == "logistic":
            # model_func = torch.sigmoid
            self.loss_func = self.logistic_loss
        else:
            # @TODO Not implement for mse now
            self.loss_func = self.mse_loss
        self.prob = cp.Problem(
            cp.Minimize(
                self.__evaluate_loss(
                    self.X,
                    self.X_prime,
                    self.y,
                    self.D,
                    self.M,
                    self.gamma,
                    self.lambda_val,
                    self.nu_inv_val,
                )
            )
        )
        self.prob_tilde = cp.Problem(
            cp.Minimize(
                self.__evaluate_loss(
                    self.X,
                    self.X_prime,
                    self.y,
                    self.D,
                    self.M_tilde,
                    self.gamma_tilde,
                    self.lambda_val,
                    self.nu_inv_val,
                )
            )
        )

    def logistic_loss(self, X, X_prime, y, M, gamma):
        X_pred = X_prime @ self.beta + M @ gamma
        log_likelihood = cp.sum(
            cp.multiply(y, X_pred)
            - cp.logistic(X_pred)
        )
        return (-log_likelihood) / (len(X_prime)) 

    def mse_loss(self, X, X_prime, y, M, gamma):
        X_pred = X_prime @ self.beta + M @ gamma
        return cp.norm2(X_pred - y) ** 2 / len(X_prime)

    def __evaluate_loss(self, X,  X_prime, y, D, M, gamma, lambda_val, nu_inv_val):
        loss_val = self.loss_func(X, X_prime, y, M, gamma)
        l2_reg_val = nu_inv_val * (
            cp.norm2(D @ self.beta - gamma) ** 2
        )
        l1_reg_val = lambda_val * (cp.norm1(gamma))
        return loss_val + l2_reg_val + l1_reg_val

    def cv_logistic_loss(self, beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde):
        def np_logistic(x):
            return np.log1p(np.exp(x))

        X_pred = X_prime @ beta + M @ gamma
        log_likelihood = np.sum(
            np.multiply(y, X_pred)
            - np_logistic(X_pred)
        )
        # X_pred_tilde = X_prime @ beta + M_tilde @ gamma_tilde
        # log_likelihood_tilde = np.sum(
        #     np.multiply(y, X_pred_tilde)
        #     - np_logistic(X_pred_tilde)
        # )
        return -(log_likelihood) / (len(X_prime))
        
    def cv_mse_loss(self, beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde):
        X_pred = X_prime @ beta + M @ gamma 
        return np.linalg.norm(X_pred - y, ord=2) ** 2 / len(X_prime)

    def cv_evaluate_loss(
        self, beta, gamma, gamma_tilde, X, y, D, M, M_tilde, lambda_val, nu_inv_val
    ):
        if beta is None or gamma is None or gamma_tilde is None:
            return np.inf
        X_prime = X - M @ D
        if self.model_type == "logistic":
            cv_loss_func = self.cv_logistic_loss
        else:
            cv_loss_func = self.cv_mse_loss
        loss_val = cv_loss_func(beta, gamma, gamma_tilde, X, X_prime, y, M, M_tilde)
        l2_reg_val = nu_inv_val * (
            np.linalg.norm(D @ beta - gamma, ord=2) ** 2
        )
        l1_reg_val = lambda_val * (
            np.linalg.norm(gamma, ord=1)
        )
        # print("CV losses", lambda_val, nu_inv_val, loss_val, l2_reg_val, l1_reg_val)
        return loss_val + l2_reg_val + l1_reg_val

    # Adam n_iters=100, lr=1
    def solve(self, lambda_val, nu_inv_val):
        self.lambda_val.value = lambda_val
        self.nu_inv_val.value = nu_inv_val
        self.prob.solve(solver=cp.MOSEK, verbose=False)
        beta_val = self.beta.value
        self.prob_tilde.solve(solver=cp.MOSEK, verbose=False)
        # print("Prob:", self.prob.value)
        # print("Solved Results", self.beta.value, self.gamma.value, self.gamma_tilde.value)
        return beta_val, self.gamma.value, self.gamma_tilde.value
