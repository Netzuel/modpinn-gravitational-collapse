"""Paper residual loss and adaptive collocation; no reference-metric feedback."""

import math
import torch
from .model import DTYPES


def make_grid(config):
    """Create the time-major paper grid, including its positive radial offset."""
    DTYPE = DTYPES[config["training_process"]["DTYPE"]]
    device = torch.device(config["training_process"]["device"])
    (t_range, x_range) = (
        config["physical"]["parameters"]["t_range"],
        config["physical"]["parameters"]["x_range"],
    )
    (N_t, N_x) = (
        int(config["physical"]["parameters"]["N_t"]),
        int(config["physical"]["parameters"]["N_x"]),
    )
    (t, x) = (
        torch.linspace(t_range[0], t_range[1], N_t),
        torch.linspace(x_range[0], x_range[1], N_x),
    )
    offset = (x.max() - x.min()) / x.shape[0]
    x += offset
    (x_mesh, t_mesh) = torch.meshgrid(x, t, indexing="xy")
    data = torch.stack((t_mesh, x_mesh), dim=-1)
    X = data.reshape(-1, 2)
    return X.to(DTYPE).to(device).requires_grad_(True)


def replace_last_nan_with_prev(x: torch.Tensor) -> torch.Tensor:
    """
    x: shape (N, 1). If x[-1, 0] is NaN, set it to x[-2, 0].
    Returns x, modified in place.
    """
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"Expected shape (N, 1), got {tuple(x.shape)}")
    if x.shape[0] < 2:
        return x
    last_is_nan = torch.isnan(x[-1, 0])
    x[-1, 0] = torch.where(last_is_nan, x[-2, 0], x[-1, 0])
    return x


def loss(model, X_r, U_0, U_b_left):
    """Compute loss with both time‐ and space‐causality enforcement, fully vectorized."""
    model.U_0 = U_0
    model.U_b_left = U_b_left
    (t, r) = (X_r[:, 0:1], X_r[:, 1:2])
    (φ, α, Q) = model(torch.cat((t, r), dim=1))
    Q = torch.clamp(Q, min=0, max=0.999999)
    lnα = torch.log(α)
    φ_bL_true = U_b_left[:, 0:1]
    α_bL_true = U_b_left[:, 1:2]
    Q_bL_true = U_b_left[:, 2:3]
    φ_bL_pred = φ.view(model.N_t, model.N_x, 1)[:, 0, :]
    α_bL_pred = α.view(model.N_t, model.N_x, 1)[:, 0, :]
    Q_bL_pred = Q.view(model.N_t, model.N_x, 1)[:, 0, :]
    L_bL = (
        (φ_bL_true - φ_bL_pred).pow(2)
        + (α_bL_true - α_bL_pred).pow(2)
        + (Q_bL_true - Q_bL_pred).pow(2)
    )
    φ_0_true = replace_last_nan_with_prev(U_0[:, 0:1])
    α_0_true = replace_last_nan_with_prev(U_0[:, 1:2])
    Q_0_true = replace_last_nan_with_prev(U_0[:, 2:3])
    φ_0_pred = φ.view(model.N_t, model.N_x, 1)[0]
    α_0_pred = α.view(model.N_t, model.N_x, 1)[0]
    Q_0_pred = Q.view(model.N_t, model.N_x, 1)[0]
    L_ic_φ = (φ_0_true - φ_0_pred).pow(2)
    L_ic_α = (α_0_true - α_0_pred).pow(2)
    L_ic_Q = (Q_0_true - Q_0_pred).pow(2)
    L_φ_peak = (φ_0_pred.max() - φ_0_true.max()).pow(2)
    L_ic_φ += L_φ_peak
    dφ_dt = torch.autograd.grad(φ, t, grad_outputs=torch.ones_like(φ), create_graph=True)[0]
    dφ_dr = torch.autograd.grad(φ, r, grad_outputs=torch.ones_like(φ), create_graph=True)[0]
    Φ = dφ_dr
    Π = dφ_dt / ((1 - Q).sqrt() * α)
    Π_ic = Π.view(model.N_t, model.N_x, 1)[0]
    L_ic_Π = Π_ic.pow(2)
    dlnα_dr = torch.autograd.grad(lnα, r, grad_outputs=torch.ones_like(lnα), create_graph=True)[0]
    dα_dr = torch.autograd.grad(α, r, grad_outputs=torch.ones_like(α), create_graph=True)[0]
    dQ_dr = torch.autograd.grad(Q, r, grad_outputs=torch.ones_like(Q), create_graph=True)[0]
    m = r * Q / 2
    dm_dr = torch.autograd.grad(m, r, grad_outputs=torch.ones_like(m), create_graph=True)[0]
    Z1 = math.sqrt(2 * math.pi) * r * (1 - Q).sqrt() * Φ
    Z2 = math.sqrt(2 * math.pi) * r * (1 - Q).sqrt() * Π
    dΦ_dt = torch.autograd.grad(Φ, t, grad_outputs=torch.ones_like(Φ), create_graph=True)[0]
    G = torch.autograd.grad(
        α * (1 - Q).sqrt() * Π, r, grad_outputs=torch.ones_like(Π), create_graph=True
    )[0]
    dΠ_dt = torch.autograd.grad(Π, t, grad_outputs=torch.ones_like(Π), create_graph=True)[0]
    H = (
        1
        / r**2
        * torch.autograd.grad(
            r**2 * (1 - Q).sqrt() * α * Φ, r, grad_outputs=torch.ones_like(Φ), create_graph=True
        )[0]
    )
    eq_1a = dΦ_dt - G
    eq_1b = dΠ_dt - H
    eq_2 = r * (1 - Q) * dlnα_dr - r / 2 * dQ_dr - Q
    eq_3 = dm_dr - (Z1.pow(2) + Z2.pow(2))
    R_φ = (r * dφ_dr + φ).view(model.N_t, model.N_x, 1)[:, -1, :]
    R_α = (r * dα_dr + α - 1).view(model.N_t, model.N_x, 1)[:, -1, :]
    R_Q = (r * dQ_dr + Q).view(model.N_t, model.N_x, 1)[:, -1, :]
    L_bR = R_φ.pow(2) + R_α.pow(2) + R_Q.pow(2)
    eqs = eq_1a.pow(2) + eq_1b.pow(2) + eq_2.pow(2) + eq_3.pow(2)
    eqs = eqs.view(model.N_t, model.N_x, 1)
    eqs[:, 0, 0] = model.w_boundary[0] * L_bL.squeeze()
    eqs[:, -1, 0] = model.w_boundary[1] * L_bR.squeeze()
    eqs = eqs.squeeze(-1)
    zeros_r = torch.zeros((model.N_t, 1), device=model.device)
    shifted_r = torch.cat([zeros_r, eqs[:, :-1]], dim=1)
    S_r = torch.cumsum(shifted_r, dim=1)
    w_r = torch.exp(-model.eps_spatial * S_r)
    model.w_r = w_r
    eqs_spatial = w_r * eqs
    L_ic = model.w_φ * L_ic_φ + model.w_α * L_ic_α + model.w_Q * L_ic_Q + model.w_Π * L_ic_Π
    eqs_spatial[0, :] = L_ic.squeeze()
    L_t_space = eqs_spatial.mean(dim=1, keepdim=True)
    L_t_shifted = torch.cat((torch.zeros(1, 1, device=model.device), L_t_space[:-1]), dim=0)
    L_t_cumsum = torch.cumsum(L_t_shifted, dim=0)
    w_t_tensor = torch.exp(-model.eps_causality * L_t_cumsum)
    model.w_t = w_t_tensor
    model.AUC_hist.append(torch.trapz(w_t_tensor.view(-1), dx=1 / model.N_t).item())
    L = torch.mean(w_t_tensor * L_t_space)
    model.eqs = eq_1a.pow(2) + eq_1b.pow(2) + eq_2.pow(2) + eq_3.pow(2)
    return L


def remesh(config, model, X_r):
    """
    Adaptive weighted-CDF radial remeshing, holding the initial time slice fixed.
    This version fully vectorizes over N_t time‐slices.

    Expects in `config`:
        - config["physical"]["parameters"]["domain_step"] (float >= 0)

    Expects `model` to have:
        - xmin, xmax: radial bounds from the supplied reference
        - model.eqs: sum of four squared residuals, shape [N_t * N_x, 1]
        - model.N_t, model.N_x
        - model.device
    Returns:
        new_X_r: Tensor of shape (N_t * N_x, 2).
        The supplied X_r has that same shape; its entire initial slice is fixed.
    """
    lam = config["physical"]["parameters"]["domain_step"]
    r_min = model.xmin
    r_max = model.xmax
    device = model.device
    N_t = model.N_t
    N_x = model.N_x
    with torch.no_grad():
        X_flat = X_r.detach().to(device)
        eqs_flat = model.eqs.detach().view(-1)
        X_grid = X_flat.view(N_t, N_x, 2)
        eqs_grid = eqs_flat.view(N_t, N_x)
        X_new = X_grid.clone()
        r_old = X_grid[..., 1].clone()
        L_old = eqs_grid.clone()
        (r_sorted, sort_idx) = torch.sort(r_old, dim=1)
        L_sorted = L_old.gather(1, sort_idx)
        L_min = L_sorted.min(dim=1, keepdim=True).values
        L_max = L_sorted.max(dim=1, keepdim=True).values
        diff = (L_max - L_min).clamp(min=1e-08)
        L_norm = (L_sorted - L_min) / diff
        m = 1.0 + lam * L_norm
        delta_r = r_sorted[:, 1:] - r_sorted[:, :-1]
        m_mid = 0.5 * (m[:, :-1] + m[:, 1:])
        w = m_mid * delta_r
        W = torch.zeros((N_t, N_x), dtype=r_sorted.dtype, device=device)
        W[:, 1:] = torch.cumsum(w, dim=1)
        W_total = W[:, -1].clone()
        good = W_total > 0.0
        idx = torch.arange(N_x, dtype=r_sorted.dtype, device=device)
        c = idx[None, :] / float(N_x - 1) * W_total[:, None]
        k = torch.searchsorted(W, c)
        j = (k - 1).clamp(min=0, max=N_x - 2)
        w_j = w.gather(1, j)
        dr_j = delta_r.gather(1, j)
        W_j = W.gather(1, j)
        r_j = r_sorted.gather(1, j)
        r_new_sorted = torch.zeros_like(r_sorted)
        r_new_sorted[:, 0] = r_sorted[:, 0]
        r_new_sorted[:, -1] = r_sorted[:, -1]
        if N_x > 2:
            interior = slice(1, N_x - 1)
            delta_mass = c[:, interior] - W_j[:, interior]
            denom = w_j[:, interior].clone()
            denom[denom.abs() < 1e-12] = 1e-12
            frac = delta_mass / denom
            frac = frac.clamp(min=0.0, max=1.0)
            r_j_int = r_j[:, interior]
            dr_j_int = dr_j[:, interior]
            r_new_sorted[:, interior] = r_j_int + frac * dr_j_int
        if (~good).any():
            r_new_sorted[~good, :] = r_sorted[~good, :]
        unsort_idx = sort_idx.argsort(dim=1)
        r_unsorted = r_new_sorted.gather(1, unsort_idx)
        r_clamped = torch.clamp(r_unsorted, min=r_min, max=r_max)
        r_clamped[0, :] = r_old[0, :]
        X_new[:, :, 1] = r_clamped
        new_X_flat = X_new.view(N_t * N_x, 2)
        return new_X_flat.requires_grad_(True)
