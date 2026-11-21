import torch
import torch.nn.functional as F


@torch.no_grad()
def batched_cem_vectorized(
    rollout_cost_fn,
    B,
    horizon,
    act_dim,
    device,
    low=-1.0,
    high=1.0,
    pop_size=384,
    elite_frac=0.08,
    n_iters=5,
):
    """Runs Euclidean CEM in parallel across all B evaluation episodes."""
    mean = torch.zeros((B, horizon, act_dim), device=device)
    std = torch.ones((B, horizon, act_dim), device=device) * 0.70
    n_elite = max(1, int(pop_size * elite_frac))
    batch_idx = torch.arange(B, device=device).unsqueeze(-1)

    for _ in range(n_iters):
        noise = torch.randn((B, pop_size, horizon, act_dim), device=device)
        samples = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(low, high)
        costs = rollout_cost_fn(samples)
        elite_idx = torch.topk(costs, n_elite, dim=-1, largest=False).indices
        elites = samples[batch_idx, elite_idx]
        mean = 0.75 * elites.mean(dim=1) + 0.25 * mean
        std = 0.75 * elites.std(dim=1) + 0.25 * std + 1e-3
    return mean


@torch.no_grad()
def batched_cem_simplex(
    rollout_cost_fn,
    horizon,
    act_dim,
    device,
    pop_size=384,
    elite_frac=0.10,
    n_iters=5,
):
    """Runs CEM over logits projected to simplex portfolio weights via softmax."""
    mean = torch.zeros((horizon, act_dim), device=device)
    std = torch.ones((horizon, act_dim), device=device) * 1.5
    n_elite = max(1, int(pop_size * elite_frac))

    for _ in range(n_iters):
        noise = torch.randn((pop_size, horizon, act_dim), device=device)
        logits = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-4.0, 4.0)
        weights = F.softmax(logits, dim=-1)
        costs = rollout_cost_fn(weights)
        elites = logits[torch.topk(costs, n_elite, largest=False).indices]
        mean = 0.75 * elites.mean(dim=0) + 0.25 * mean
        std = 0.75 * elites.std(dim=0) + 0.25 * std + 1e-3
    return F.softmax(mean, dim=-1)
