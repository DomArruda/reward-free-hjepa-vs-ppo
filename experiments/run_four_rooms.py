import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models import GoalPPO, train_hjepa
from src.planners import batched_cem_vectorized
from src.stats_utils import kruskal_wallis, run_posthoc_welch, safe_alexander_govern

N_SEEDS = 8
N_EVAL_EPISODES = 24
ALPHA = 0.05
JEPA_ENV_STEPS = 4096 * 96


class GPUFourRooms:
    def __init__(self, dt=0.08, damping=0.92, device=None):
        self.dt = dt
        self.damping = damping
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        walls_list = [
            [-0.03, -1.00,  0.03, -0.65],
            [-0.03, -0.25,  0.03,  0.25],
            [-0.03,  0.65,  0.03,  1.00],
            [-1.00, -0.03, -0.65,  0.03],
            [-0.25, -0.03,  0.25,  0.03],
            [ 0.65, -0.03,  1.00,  0.03],
        ]
        self.walls = torch.tensor(walls_list, dtype=torch.float32, device=self.device)

    def in_wall(self, x, y):
        out_bounds = (x.abs() > 0.98) | (y.abs() > 0.98)
        in_w = (
            (x.unsqueeze(-1) >= self.walls[:, 0]) &
            (x.unsqueeze(-1) <= self.walls[:, 2]) &
            (y.unsqueeze(-1) >= self.walls[:, 1]) &
            (y.unsqueeze(-1) <= self.walls[:, 3])
        ).any(dim=-1)
        return out_bounds | in_w

    @torch.no_grad()
    def sample_free_states(self, batch_size):
        states = torch.zeros((batch_size, 4), device=self.device)
        remaining = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        while remaining.any():
            n = remaining.sum().item()
            xy = torch.empty((n, 2), device=self.device).uniform_(-0.9, 0.9)
            bad = self.in_wall(xy[:, 0], xy[:, 1])
            valid_idx = torch.where(remaining)[0][~bad]
            states[valid_idx, :2] = xy[~bad]
            states[valid_idx, 2:] = torch.empty((len(valid_idx), 2), device=self.device).uniform_(-0.2, 0.2)
            remaining[valid_idx] = False
        return states

    @torch.no_grad()
    def step(self, s, a):
        a = a.clamp(-1.0, 1.0)
        x, y, vx, vy = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        nvx = self.damping * vx + a[:, 0] * self.dt * 2.5
        nvy = self.damping * vy + a[:, 1] * self.dt * 2.5
        nx = x + nvx * self.dt
        ny = y + nvy * self.dt

        hit_x = self.in_wall(nx, y)
        nx = torch.where(hit_x, x, nx)
        nvx = torch.where(hit_x, -0.3 * nvx, nvx)

        hit_y = self.in_wall(x, ny)
        ny = torch.where(hit_y, y, ny)
        nvy = torch.where(hit_y, -0.3 * nvy, nvy)

        hit_xy = self.in_wall(nx, ny)
        nx = torch.where(hit_xy, x, nx)
        ny = torch.where(hit_xy, y, ny)
        nvx = torch.where(hit_xy, -0.3 * nvx, nvx)
        nvy = torch.where(hit_xy, -0.3 * nvy, nvy)

        return torch.stack([nx, ny, nvx, nvy], dim=-1)

    @torch.no_grad()
    def collect_rollouts(self, n_traj=4096, horizon=96):
        states = torch.zeros((n_traj, horizon + 1, 4), device=self.device)
        actions = torch.zeros((n_traj, horizon, 2), device=self.device)
        s = self.sample_free_states(n_traj)
        states[:, 0] = s
        a = torch.empty((n_traj, 2), device=self.device).uniform_(-1.0, 1.0)
        for t in range(horizon):
            if t % 6 == 0:
                a = torch.empty((n_traj, 2), device=self.device).uniform_(-1.0, 1.0)
            else:
                a = (0.90 * a + 0.15 * torch.randn_like(a)).clamp(-1.0, 1.0)
            actions[:, t] = a
            s = self.step(s, a)
            states[:, t + 1] = s
        return states, actions


@torch.no_grad()
def measure_action_sensitivity(model, s_probe):
    """Causal mechanism check: ||z_hat(s, +1) - z_hat(s, -1)||^2 across probe states."""
    device = s_probe.device
    z = model.enc(s_probe)
    a_pos = torch.ones((s_probe.shape[0], 2), device=device)
    a_neg = -torch.ones((s_probe.shape[0], 2), device=device)
    sens_l1 = F.mse_loss(model.step1(z, a_pos), model.step1(z, a_neg)).item()
    sens_l2 = F.mse_loss(model.step2(z, a_pos), model.step2(z, a_neg)).item()
    return sens_l1, sens_l2


@torch.no_grad()
def eval_jepa_vectorized(env, model, s_starts, s_goals, hierarchical=True, macro_horizon=5, flat_horizon=2, max_steps=150):
    device = s_starts.device
    B = s_starts.shape[0]
    s_curr = s_starts.clone()
    z_goal = model.enc(s_goals)
    z_subgoal = z_goal.clone()
    done = torch.zeros(B, dtype=torch.bool, device=device)
    traj = [s_curr[0].cpu().numpy()]

    for step in range(max_steps):
        reached = torch.norm(s_curr[:, :2] - s_goals[:, :2], dim=-1) < 0.12
        done = done | reached
        if done.all():
            break

        z_curr = model.enc(s_curr)

        if hierarchical:
            if step % 3 == 0:
                def cost_l2(macro_seqs):
                    _, P, H, _ = macro_seqs.shape
                    z = z_curr.unsqueeze(1).expand(B, P, -1)
                    zg = z_goal.unsqueeze(1)
                    traj_err = 0.0
                    for h in range(H):
                        z = model.step2(z, macro_seqs[:, :, h, :])
                        traj_err = traj_err + ((z - zg) ** 2).mean(dim=-1)
                    return ((z - zg) ** 2).mean(dim=-1) + 0.20 * (traj_err / H)

                best_macro = batched_cem_vectorized(cost_l2, B, macro_horizon, 2, device=device, pop_size=384, n_iters=5)
                z_subgoal = model.step2(z_curr, best_macro[:, 0, :])

            def cost_l1(prim_seqs):
                _, P, K, _ = prim_seqs.shape
                z = z_curr.unsqueeze(1).expand(B, P, -1)
                for k_step in range(K):
                    z = model.step1(z, prim_seqs[:, :, k_step, :])
                sub_err = ((z - z_subgoal.unsqueeze(1)) ** 2).mean(dim=-1)
                goal_err = ((z - z_goal.unsqueeze(1)) ** 2).mean(dim=-1)
                return 0.55 * sub_err + 0.45 * goal_err

            best_prim = batched_cem_vectorized(cost_l1, B, model.k, 2, device=device, pop_size=256, n_iters=4)
        else:
            def cost_flat(prim_seqs):
                _, P, K, _ = prim_seqs.shape
                z = z_curr.unsqueeze(1).expand(B, P, -1)
                for k_step in range(K):
                    z = model.step1(z, prim_seqs[:, :, k_step, :])
                return ((z - z_goal.unsqueeze(1)) ** 2).mean(dim=-1)

            best_prim = batched_cem_vectorized(cost_flat, B, flat_horizon, 2, device=device, pop_size=384, n_iters=5)

        s_next = env.step(s_curr, best_prim[:, 0, :])
        s_curr = torch.where(done.unsqueeze(-1), s_curr, s_next)
        traj.append(s_curr[0].cpu().numpy())

    failure_rate = (1.0 - done.float().mean().item()) * 100.0
    return failure_rate, np.array(traj)


def train_and_eval_ppo(env, s_starts, s_goals, num_envs=512, rollout_len=150, updates=50):
    device = env.device
    t_tr0 = time.time()
    ppo = GoalPPO().to(device)
    opt = torch.optim.Adam(ppo.parameters(), lr=3e-3)

    for _ in range(updates):
        s = env.sample_free_states(num_envs)
        g = env.sample_free_states(num_envs)
        obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
        for _ in range(rollout_len):
            obs = torch.cat([s, g], dim=-1)
            with torch.no_grad():
                act, logp, _, val = ppo.get_action_and_value(obs)
            s_next = env.step(s, act)
            dist = torch.norm(s_next[:, :2] - g[:, :2], dim=-1)
            rew = -dist + 1.0 * (dist < 0.12).float()
            obs_buf.append(obs)
            act_buf.append(act)
            logp_buf.append(logp)
            rew_buf.append(rew)
            val_buf.append(val)
            s = s_next

        returns = torch.zeros((rollout_len, num_envs), device=device)
        R = torch.zeros(num_envs, device=device)
        for t in reversed(range(rollout_len)):
            R = rew_buf[t] + 0.99 * R
            returns[t] = R

        b_obs = torch.stack(obs_buf).reshape(-1, 8)
        b_act = torch.stack(act_buf).reshape(-1, 2)
        b_logp = torch.stack(logp_buf).reshape(-1)
        b_ret = returns.reshape(-1)
        b_adv = b_ret - torch.stack(val_buf).reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(4):
            _, new_logp, entropy, new_val = ppo.get_action_and_value(b_obs, b_act)
            ratio = torch.exp(new_logp - b_logp)
            loss = (
                -torch.min(ratio * b_adv, torch.clamp(ratio, 0.8, 1.2) * b_adv).mean()
                + 0.5 * F.mse_loss(new_val, b_ret)
                - 0.01 * entropy.mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    t_train = time.time() - t_tr0

    t_ev0 = time.time()
    with torch.no_grad():
        s = s_starts.clone()
        B = s.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=device)
        traj = [s[0].cpu().numpy()]
        for _ in range(rollout_len):
            act = ppo.actor_mean(torch.cat([s, s_goals], dim=-1))
            s = env.step(s, act)
            traj.append(s[0].cpu().numpy())
            done = done | (torch.norm(s[:, :2] - s_goals[:, :2], dim=-1) < 0.12)
    t_eval = time.time() - t_ev0

    env_steps = num_envs * rollout_len * updates
    return (1.0 - done.float().mean().item()) * 100.0, np.array(traj), t_train, t_eval, env_steps


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Upgraded Four-Rooms Study (Vectorized CEM + Robust Stats) on: {device}")

    conditions = [
        "H-JEPA + SIGReg + InvDyn (Full)",
        "H-JEPA + SIGReg (No InvDyn)",
        "Flat 1-Level JEPA + InvDyn (Ablation)",
        "PPO Actor-Critic (RL Baseline)",
    ]
    results = {c: [] for c in conditions}
    train_times = {c: [] for c in conditions}
    eval_times = {c: [] for c in conditions}
    wall_eff = {c: [] for c in conditions}
    sample_eff = {c: [] for c in conditions}
    sens_logs = {"With InvDyn (L1/L2)": [], "No InvDyn (L1/L2)": []}

    env = GPUFourRooms(device=device)
    t_exp = time.time()

    for seed in range(N_SEEDS):
        torch.manual_seed(100 + seed)
        np.random.seed(100 + seed)

        s_starts = torch.tensor(
            [[-0.70, -0.70, 0.0, 0.0]] * (N_EVAL_EPISODES // 2)
            + [[-0.70, 0.70, 0.0, 0.0]] * (N_EVAL_EPISODES // 2),
            device=device,
        )
        s_starts[:, :2] += torch.empty((N_EVAL_EPISODES, 2), device=device).uniform_(-0.08, 0.08)
        s_goals = torch.tensor(
            [[0.70, 0.70, 0.0, 0.0]] * (N_EVAL_EPISODES // 2)
            + [[0.70, -0.70, 0.0, 0.0]] * (N_EVAL_EPISODES // 2),
            device=device,
        )
        s_goals[:, :2] += torch.empty((N_EVAL_EPISODES, 2), device=device).uniform_(-0.08, 0.08)
        s_probe = env.sample_free_states(256)

        t0 = time.time()
        states, actions = env.collect_rollouts(n_traj=4096, horizon=96)
        t_data = time.time() - t0

        t0 = time.time()
        model_full = train_hjepa(states, actions, use_inv_head=True, macro_stride=6, rollout_h=5, epochs=280, batch_size=512)
        t_tr_full = (time.time() - t0) + t_data
        s1_f, s2_f = measure_action_sensitivity(model_full, s_probe)
        sens_logs["With InvDyn (L1/L2)"].append((s1_f, s2_f))

        t0 = time.time()
        fail_full, _ = eval_jepa_vectorized(env, model_full, s_starts, s_goals, hierarchical=True)
        t_ev_full = time.time() - t0

        t0 = time.time()
        fail_flat, _ = eval_jepa_vectorized(env, model_full, s_starts, s_goals, hierarchical=False, flat_horizon=2)
        t_ev_flat = time.time() - t0

        t0 = time.time()
        model_no_inv = train_hjepa(states, actions, use_inv_head=False, macro_stride=6, rollout_h=5, epochs=280, batch_size=512)
        t_tr_noinv = (time.time() - t0) + t_data
        s1_n, s2_n = measure_action_sensitivity(model_no_inv, s_probe)
        sens_logs["No InvDyn (L1/L2)"].append((s1_n, s2_n))

        t0 = time.time()
        fail_no_inv, _ = eval_jepa_vectorized(env, model_no_inv, s_starts, s_goals, hierarchical=True)
        t_ev_noinv = time.time() - t0

        fail_ppo, _, t_tr_ppo, t_ev_ppo, ppo_steps = train_and_eval_ppo(env, s_starts, s_goals)

        for name, fl, tr_t, ev_t, steps in [
            ("H-JEPA + SIGReg + InvDyn (Full)", fail_full, t_tr_full, t_ev_full, JEPA_ENV_STEPS),
            ("H-JEPA + SIGReg (No InvDyn)", fail_no_inv, t_tr_noinv, t_ev_noinv, JEPA_ENV_STEPS),
            ("Flat 1-Level JEPA + InvDyn (Ablation)", fail_flat, t_tr_full, t_ev_flat, JEPA_ENV_STEPS),
            ("PPO Actor-Critic (RL Baseline)", fail_ppo, t_tr_ppo, t_ev_ppo, ppo_steps),
        ]:
            results[name].append(fl)
            train_times[name].append(tr_t)
            eval_times[name].append(ev_t)
            wall_eff[name].append((100.0 - fl) / (tr_t + ev_t))
            sample_eff[name].append((100.0 - fl) / (steps / 1e5))

        print(
            f"Seed {seed + 1}/{N_SEEDS} | Fail -> Full: {fail_full:5.1f}% | "
            f"No-Inv: {fail_no_inv:5.1f}% (Sens:{s2_n:.4f}) | "
            f"Flat: {fail_flat:5.1f}% | PPO: {fail_ppo:5.1f}% | InvDyn Sens:{s2_f:.4f}"
        )

    groups_fail = [np.array(results[c]) for c in conditions]
    groups_wall = [np.array(wall_eff[c]) for c in conditions]
    groups_samp = [np.array(sample_eff[c]) for c in conditions]

    kw_f, p_kw_f = kruskal_wallis(groups_fail)
    ag_f, p_ag_f = safe_alexander_govern(groups_fail)
    kw_w, p_kw_w = kruskal_wallis(groups_wall)
    ag_w, p_ag_w = safe_alexander_govern(groups_wall)
    kw_s, p_kw_s = kruskal_wallis(groups_samp)
    ag_s, p_ag_s = safe_alexander_govern(groups_samp)

    print("\n" + "=" * 96)
    print(f"1. CAUSAL MECHANISM CHECK: ACTION SENSITIVITY ||z_hat(s, +1) - z_hat(s, -1)||² (N = {N_SEEDS} SEEDS)")
    print("-" * 96)
    w_inv = np.array(sens_logs["With InvDyn (L1/L2)"])
    n_inv = np.array(sens_logs["No InvDyn (L1/L2)"])
    print(f"  With Inverse Dynamics Head   | Level-1 Sens: {w_inv[:, 0].mean():.4f} ± {w_inv[:, 0].std():.4f} | Level-2 Sens: {w_inv[:, 1].mean():.4f} ± {w_inv[:, 1].std():.4f}")
    print(f"  Without Inverse Dynamics     | Level-1 Sens: {n_inv[:, 0].mean():.4f} ± {n_inv[:, 0].std():.4f} | Level-2 Sens: {n_inv[:, 1].mean():.4f} ± {n_inv[:, 1].std():.4f}")

    print("\n" + "=" * 96)
    print(f"2. FULL COMPUTE, SAMPLE EFFICIENCY & FAILURE RATE BREAKDOWN (N = {N_SEEDS} SEEDS)")
    print("-" * 96)
    print(f"{'Condition':<38} | {'Fail %':<10} | {'Train(s)':<8} | {'Eval(s)':<7} | {'%/Sec':<8} | {'%/100k Steps'}")
    print("-" * 96)
    for c in conditions:
        print(
            f"{c:<38} | {np.mean(results[c]):5.1f}±{np.std(results[c], ddof=1):4.1f} | "
            f"{np.mean(train_times[c]):>7.2f}s | {np.mean(eval_times[c]):>6.2f}s | "
            f"{np.mean(wall_eff[c]):>7.2f} | {np.mean(sample_eff[c]):>10.2f}"
        )

    print("\n" + "=" * 96)
    print("3. ROBUST OMNIBUS TESTS (HETEROSCEDASTIC ALEXANDER-GOVERN & NON-PARAMETRIC KRUSKAL-WALLIS)")
    print("-" * 96)
    print(f"  Failure Rate (%)             | Alexander-Govern = {ag_f:7.2f} (p = {p_ag_f:.3e}) | Kruskal-Wallis H = {kw_f:6.2f} (p = {p_kw_f:.3e})")
    print(f"  Wall-Clock Eff (%/Total Sec) | Alexander-Govern = {ag_w:7.2f} (p = {p_ag_w:.3e}) | Kruskal-Wallis H = {kw_w:6.2f} (p = {p_kw_w:.3e})")
    print(f"  Sample Eff (%/100k Steps)    | Alexander-Govern = {ag_s:7.2f} (p = {p_ag_s:.3e}) | Kruskal-Wallis H = {kw_s:6.2f} (p = {p_kw_s:.3e})")

    print("\n" + "=" * 96)
    print("4. TWO-TAILED WELCH POST-HOC CONTRASTS VS. FULL H-JEPA (HOLM-BONFERRONI CORRECTED)")
    print("-" * 96)
    base_f = np.array(results["H-JEPA + SIGReg + InvDyn (Full)"])
    comp_f = {c: np.array(results[c]) for c in conditions[1:]}
    rows_f, adj_f = run_posthoc_welch(base_f, comp_f)
    for i, (name, diff, t_s, _) in enumerate(rows_f):
        sig = "SIGNIFICANT *" if adj_f[i] < ALPHA else "ns"
        print(f"  [Fail %]   vs. {name:<37} | Δ = {diff:+6.2f}% | t = {t_s:6.2f} | p_adj = {adj_f[i]:.3e} ({sig})")
    print("=" * 96)
    print(f"Total experiment wall-clock time: {time.time() - t_exp:.1f}s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    short_labels = ["H-JEPA\n+InvDyn", "H-JEPA\nNo-InvDyn", "Flat JEPA\n(Ablation)", "PPO RL\nBaseline"]
    colors = ["#2980b9", "#95a5a6", "#8e44ad", "#e67e22"]

    for ax, g_data, ylabel, title in [
        (axes[0], groups_fail, "Failure Rate (%) — Lower is Better", f"Failure Rate (KW p={p_kw_f:.1e})"),
        (axes[1], groups_wall, "Success % / Total Sec — Higher is Better", f"Wall-Clock Efficiency (KW p={p_kw_w:.1e})"),
        (axes[2], groups_samp, "Success % / 100k Steps — Higher is Better", f"Sample Efficiency (KW p={p_kw_s:.1e})"),
    ]:
        bp = ax.boxplot(g_data, patch_artist=True, widths=0.45, medianprops=dict(color="black", lw=2))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        for idx, g in enumerate(g_data, 1):
            ax.scatter(np.full_like(g, idx) + np.random.normal(0, 0.04, len(g)), g, color="black", s=24, zorder=4)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
