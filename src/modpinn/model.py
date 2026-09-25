"""ModPINN architecture used by the paper's training code.

The RBF bank intentionally receives physical coordinates, matching the retained
implementation and weights. Coordinate normalization applies to the other features.
"""

import torch
from torch import nn
from torch.nn import functional as F

DTYPES = {"torch.float32": torch.float32, "torch.float64": torch.float64}


class TrainableSigmoid(nn.Module):
    """
    Trainable Sigmoid:
      f(x) = sigmoid(w1 * x + w2)
    where:
      - w1 controls the “steepness” (slope) of the sigmoid,
      - w2 is a learnable bias inside the sigmoid argument.
    """

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sigmoid(self.w1 * x + self.w2)


class TrainableTanh(nn.Module):
    """
    Trainable Tanh:
      f(x) = tanh(w1 * x + w2)
    """

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.tanh(self.w1 * x + self.w2)


class TanhWarp1D(nn.Module):
    """
    g(x) = a * x + b + Σ_{k=1..K} c_k * tanh( s_k * (x - μ_k) )
    - Smooth, learnable, and well-behaved for higher-order autodiff.
    - We apply it separately to t and r (after simple normalization).
    """

    def __init__(
        self, K=8, init_range=1.0, dtype=torch.float32, device: str | torch.device = "cpu"
    ):
        super().__init__()
        self.K = K
        self.register_buffer("one", torch.tensor(1.0, dtype=dtype, device=device))
        self.a = nn.Parameter(torch.ones(1, dtype=dtype, device=device))
        self.b = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        self.mu = nn.Parameter(
            torch.linspace(-init_range, init_range, K, dtype=dtype, device=device)
        )
        self.log_s = nn.Parameter(torch.zeros(K, dtype=dtype, device=device))
        self.c = nn.Parameter(torch.zeros(K, dtype=dtype, device=device))

    def forward(self, x):
        xk = x - self.mu.view(1, -1)
        s = F.softplus(self.log_s) + 0.001
        bumps = torch.tanh(s * xk) * self.c.view(1, -1)
        return self.a * x + self.b + bumps.sum(dim=1, keepdim=True)


class QResLayer_noact(nn.Module):
    """Compute (W1 x + b) * (1 + W2 x); activation is applied separately."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.quad = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        lin_out = self.lin(x)
        quad_term = self.quad(x) * lin_out
        return lin_out + quad_term


class LearnableRBFEmbedding(nn.Module):
    """
    Maps x = [t, r] to a richer feature vector using:
            - normalized coords (identity)
            - smooth tanh-bump warps per dim
            - polynomial features (t, r, t^2, r^2, t*r)
            - Gaussian RBFs with learnable centers & anisotropic lengthscales
    Then projects to 'emb_dim'.
    """

    def __init__(
        self,
        emb_dim=64,
        n_rbf=32,
        t_bounds=(0.0, 10.0),
        r_bounds=(0.0, 10.0),
        dtype=torch.float32,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_rbf = n_rbf
        (t_lo, t_hi) = t_bounds
        (r_lo, r_hi) = r_bounds
        t_center = 0.5 * (t_hi + t_lo)
        t_scale = 0.5 * (t_hi - t_lo) + 1e-08
        r_center = 0.5 * (r_hi + r_lo)
        r_scale = 0.5 * (r_hi - r_lo) + 1e-08
        self.register_buffer("t_center", torch.tensor(t_center, dtype=dtype, device=device))
        self.register_buffer("t_scale", torch.tensor(t_scale, dtype=dtype, device=device))
        self.register_buffer("r_center", torch.tensor(r_center, dtype=dtype, device=device))
        self.register_buffer("r_scale", torch.tensor(r_scale, dtype=dtype, device=device))
        self.warp_t = TanhWarp1D(K=32, dtype=dtype, device=device)
        self.warp_r = TanhWarp1D(K=32, dtype=dtype, device=device)
        self.centers = nn.Parameter(torch.rand(n_rbf, 2, dtype=dtype, device=device) * 2.0 - 1.0)
        self.log_ls = nn.Parameter(torch.zeros(n_rbf, 2, dtype=dtype, device=device))
        feat_dim = 7 + n_rbf
        self.proj = QResLayer_noact(feat_dim, emb_dim)

    def _rbf(self, x):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        ls = F.softplus(self.log_ls) + 0.001
        z = diff / ls.unsqueeze(0)
        sq = (z**2).sum(dim=2)
        return torch.exp(-0.5 * sq)

    def forward(self, x):
        t = (x[:, 0:1] - self.t_center) / self.t_scale
        r = (x[:, 1:2] - self.r_center) / self.r_scale
        x_n = torch.cat([t, r], dim=1)
        (t_n, r_n) = (x_n[:, 0:1], x_n[:, 1:2])
        t_w = self.warp_t(t_n)
        r_w = self.warp_r(r_n)
        t2 = t_n * t_n
        r2 = r_n * r_n
        tr = t_n * r_n
        rbf = self._rbf(x)
        feats = torch.cat([t_n, r_n, t_w, r_w, t2, r2, tr, rbf], dim=1)
        return self.proj(feats)


class ModPINN(nn.Module):
    """Coordinate embedding and quadratic layers for spherical scalar-field collapse."""

    N_t: int
    N_x: int
    AUC_hist: list[float]
    xmin: float
    xmax: float

    def __init__(self, config):
        super().__init__()
        self.config = config
        general = config["neural_parameters"]["general_parameters"]
        physical = config["physical"]["parameters"]
        weights = config["neural_parameters"]["loss_function_parameters"]
        self.number_hidden = general["number_hidden"]
        self.number_neurons = general["number_neurons"]
        self.num_outputs = general["num_outputs"]
        self.eps_causality = physical["eps_causality"]
        self.eps_spatial = physical["eps_spatial"]
        self.DTYPE = DTYPES[config["training_process"]["DTYPE"]]
        self.device = torch.device(config["training_process"]["device"])
        self.w_φ, self.w_α, self.w_Q, self.w_Π = weights["w_initial"]
        self.w_boundary = weights["w_boundary"]
        # Preserve parameter registration order for retained SOAP trajectories.
        self.act_hidden = nn.ModuleList()
        DTYPE = self.DTYPE
        device = self.device
        (t_lo, t_hi) = self.config["physical"]["parameters"]["t_range"]
        (r_lo, r_hi) = self.config["physical"]["parameters"]["x_range"]
        self.embedding_dim = 64
        self.embed = LearnableRBFEmbedding(
            emb_dim=self.embedding_dim,
            n_rbf=32,
            t_bounds=(float(t_lo), float(t_hi)),
            r_bounds=(float(r_lo), float(r_hi)),
            dtype=DTYPE,
            device=device,
        ).to(device)
        self.dense_layers = []
        self.dense_layers.append(
            QResLayer_noact(self.embedding_dim, self.number_neurons).to(device)
        )
        for _ in range(self.number_hidden):
            layer = QResLayer_noact(self.number_neurons, self.number_neurons).to(device)
            self.dense_layers.append(layer)
        layer_final = QResLayer_noact(self.number_neurons, self.num_outputs).to(device)
        self.dense_layers.append(layer_final)
        self.params_hidden = nn.ModuleList(self.dense_layers)
        self.act_hidden = nn.ModuleList(
            [TrainableTanh().to(self.device) for _ in range(len(self.dense_layers) - 1)]
        )
        self.act_α = TrainableSigmoid().to(self.device)
        self.act_Q = TrainableSigmoid().to(self.device)

    def forward(self, x):
        """
        x: (N,2) with columns [t, r].
        Returns: φ, α, Q each (N,1)
        """
        x = self.embed(x)
        for layer, act in zip(self.dense_layers[:-1], self.act_hidden):
            x = act(layer(x))
        x = self.dense_layers[-1](x)
        φ = x[:, 0:1]
        α = self.act_α(x[:, 1:2])
        Q = self.act_Q(x[:, 2:3])
        return (φ, α, Q)
