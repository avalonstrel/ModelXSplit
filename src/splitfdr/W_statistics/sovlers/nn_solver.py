import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetworkModel(nn.Module):
    def __init__(self, model_type, X, y, D, M, M_tilde):
        super().__init__()
        self.model_type = model_type
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.D = torch.tensor(D, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.M_tilde = torch.tensor(M_tilde, dtype=torch.float32)
        self.X_prime = self.X - self.M @ self.D
        
        # self.X_prime2 = self.X_prime[300:] 
        # self.M2 = self.M[300:] 
        # self.M_tilde2 = self.M_tilde[300:] 

        # self.X_prime = self.X_prime[:300]
        # self.M = self.M[:300]
        # self.M_tilde = self.M_tilde[:300]
        # self.y = self.y[:300]

        self.input_dim = self.X_prime.shape[1] + self.M.shape[1] + self.M_tilde.shape[1]
        self.neural_net = self.build_neural_network(self.input_dim)
        # Placeholders for regularization hyperparameters
        self.lambda_ = 0.0
        self.nu_inv_ = 0.0
        

    def build_neural_network(self, input_dim):
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) if self.model_type == "linear" else nn.Linear(64, 1),
            nn.Sigmoid() if self.model_type == "logistic" else nn.Identity()
        )
        return model

    def compute_importances(self):
        # Compute mean absolute gradient per feature for X_prime, M, and M_tilde
        X_prime = self.X_prime.clone().detach().requires_grad_(True)
        M = self.M.clone().detach().requires_grad_(True)
        M_tilde = self.M_tilde.clone().detach().requires_grad_(True)
        combined_input = torch.hstack([
            X_prime,
            M,
            M_tilde
        ])
        predictions = self.neural_net(combined_input)
        grads = torch.autograd.grad(
            outputs=predictions,
            inputs=[X_prime, M, M_tilde],
            grad_outputs=torch.ones_like(predictions),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        imp_Xprime = grads[0].mean(dim=0)
        imp_M = grads[1].mean(dim=0)
        imp_Mtilde = grads[2].mean(dim=0)
        return imp_Xprime, imp_M, imp_Mtilde

    def compute_loss(self):
        # Enable gradient tracking for X_prime, M, and M_tilde
        imp_Xprime, imp_M, imp_Mtilde = self.compute_importances()

        # Prediction loss
        X_prime = self.X_prime.clone().detach()
        combined_input = torch.hstack([
            X_prime,
            self.M,
            self.M_tilde
        ])
        predictions = self.neural_net(combined_input)
        if self.model_type == "logistic":
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        prediction_loss = criterion(predictions, self.y)

        # Regularization terms (L2 on importance)
        diff_M = torch.matmul(imp_Xprime, self.D.T) - imp_M
        diff_Mtilde = torch.matmul(imp_Xprime, self.D.T) - imp_Mtilde
        # self.nu_inv_
        l2_reg = self.nu_inv_ * (torch.norm(diff_M, p=2)**2 + torch.norm(diff_Mtilde, p=2)**2)

        # L1 and L2 penalties on all importances
        # l2_grad_penalty = self.nu_inv_ * (torch.norm(imp_Xprime, p=2)**2 + torch.norm(imp_M, p=2)**2 + torch.norm(imp_Mtilde, p=2)**2)
        # self.lambda_
        l1_grad_penalty = self.lambda_ * (torch.norm(imp_Xprime, p=1) + torch.norm(imp_M, p=1) + torch.norm(imp_Mtilde, p=1))

        total_loss = prediction_loss + l1_grad_penalty + l2_reg
        return total_loss

    def train_model(self, epochs=10, lr=0.001):
        optimizer = optim.Adam(self.neural_net.parameters(), lr=lr, weight_decay=1e-3)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def solve(self, lambda_val, nu_inv_val, epochs=100, lr=0.02):
        self.lambda_ = lambda_val
        self.nu_inv_ = nu_inv_val
        
        self.train_model(epochs, lr)
        # Compute feature importances and split by section
        imp_Xprime, imp_M, imp_Mtilde = self.compute_importances()
        return (
            imp_Xprime.detach().cpu().numpy(),
            imp_M.detach().cpu().numpy(),
            imp_Mtilde.detach().cpu().numpy()
        )

    def cv_evaluate_loss(
        self, beta_imp, gamma_imp, gamma_tilde_imp, X, y, D, M, M_tilde, lambda_val=None, nu_inv_val=None
    ):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            M = torch.tensor(M, dtype=torch.float32)
            M_tilde = torch.tensor(M_tilde, dtype=torch.float32)
            X_prime = X - M @ self.D
            combined_input = torch.hstack([
                X_prime,
                M,
                M_tilde
            ])
            predictions = self.neural_net(combined_input)
            if self.model_type == "logistic":
                criterion = nn.BCELoss()
            else:
                criterion = nn.MSELoss()
            prediction_loss = criterion(predictions, y)
        return prediction_loss