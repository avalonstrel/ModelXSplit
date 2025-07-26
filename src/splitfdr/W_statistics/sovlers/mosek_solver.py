import numpy as np
import mosek
from mosek.fusion import Model, Expr, Domain, ObjectiveSense
import mosek.fusion.pythonic

def softplus(M, t, u):
    n = t.getShape()[0]
    z1 = M.variable(n)
    z2 = M.variable(n)
    M.constraint(z1 + z2 == 1)
    M.constraint(Expr.hstack(z1, Expr.constTerm(n, 1.0), u - t) == Domain.inPExpCone())
    M.constraint(Expr.hstack(z2, Expr.constTerm(n, 1.0), -t) == Domain.inPExpCone())
    # M.set(
    #     ,
    #     Expr.hstack(z1, Expr.constTerm(n, 1.0), u - t) == Domain.inPExpCone(),
    #     Expr.hstack(z2, Expr.constTerm(n, 1.0), -t) == Domain.inPExpCone(),
    # )


def mse(M, t, u):
    t = M.variable()
    M.constraint(Expr.vstack(t, u) == Domain.inQCone())


class ModelXMosekSolver:
    def __init__(self, model_type, X, y, D, M, M_tilde) -> None:
        self.model_type = model_type
        self.X, self.y = X, y
        self.D = D
        self.M, self.M_tilde = M, M_tilde
        self.X_prime = self.X - self.M @ self.D
        self.model_type = model_type

        assert model_type in [
            "logistic",
            "linear",
        ], f"Model type:{model_type} not supported."

        self.prob = self.__init_model(
            self.X,
            self.X_prime,
            self.y,
            self.D,
            self.M,
            self.M_tilde,
        )

    def __init_model(
        self,
        X,
        X_prime,
        y,
        D,
        M,
        M_tilde,
    ):
        model = Model()
        n, m = M.shape
        p = X.shape[1]
        self.beta = model.variable("beta", p)
        self.gamma = model.variable("gamma", m)
        self.gamma_tilde = model.variable("gamma_tilde", m)
        self.lambda_ = model.parameter("lambda", 1)
        self.nu_inv_ = model.parameter("nu_inv", 1)

        # Logistic / MSE loss
        t = model.variable("t", n)
        t_tilde = model.variable("t_tilde", n)
        # print(X.shape, X_prime.shape, p, m, self.beta)
        # print(Expr.mul(X_prime, self.beta))
        # print(M @ self.gamma)
        if self.model_type == "logistic":
            # logistic
            signs = list(map(lambda y: -1.0 if y == 1 else 1.0, y))
            softplus(model, t, Expr.mulElm(Expr.add(Expr.mul(X_prime, self.beta), Expr.mul(M, self.gamma)), signs))
            softplus(
                model,
                t_tilde,
                Expr.mulElm(Expr.add(Expr.mul(X_prime, self.beta), Expr.mul(M_tilde, self.gamma_tilde)), signs))
            # mse
            mse(model, t, Expr.add(Expr.mul(X_prime, self.beta), Expr.mul(M, self.gamma)) - y)
            mse(model, t, Expr.add(Expr.mul(X_prime, self.beta), Expr.mul(M_tilde, self.gamma_tilde)) - y)

        # p_i >= |x_i|, i=1..n
        p = model.variable(m)
        p_tilde = model.variable(m)
        model.constraint(Expr.hstack(p, self.gamma) == Domain.inQCone())
        model.constraint(Expr.hstack(p_tilde, self.gamma_tilde) == Domain.inQCone())

        # q >= |x|_2
        q = model.variable()
        q_tilde = model.variable()
        model.constraint(Expr.vstack(q, Expr.add(Expr.mul(D, self.beta), -self.gamma)) == Domain.inQCone())
        model.constraint(
            Expr.vstack(q_tilde, Expr.add(Expr.mul(D, self.beta), -self.gamma_tilde)) == Domain.inQCone()
        )

        # t + lambda*l1 + nu_inv * l2
        obj = (
            Expr.sum(t)
            + Expr.sum(t_tilde)
            + self.lambda_ * Expr.add(Expr.sum(p), Expr.sum(p_tilde))
            + self.nu_inv_ * Expr.add(Expr.sum(q), Expr.sum(q_tilde))
        )
        model.objective(ObjectiveSense.Minimize, obj)

        return model

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
        return loss_val + l2_reg_val + l1_reg_val

    def solve(self, lambda_val, nu_inv_val):
        self.lambda_.setValue(lambda_val)
        self.nu_inv_.setValue(nu_inv_val)
        self.prob.solve()
        return self.beta.level(), self.gamma.level(), self.gamma_tilde.level()
