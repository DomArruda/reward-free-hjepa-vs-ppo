import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf

from src.models import PortfolioPPO, train_hjepa
from src.planners import batched_cem_simplex
from src.stats_utils import kruskal_wallis

TICKERS = ["SPY", "TLT", "GLD", "XLE", "SHY"]
N_ASSETS = len(TICKERS)
N_SEEDS = 8


def load_rate_shock_data(device):
    """Download ETF prices and build strictly lagged (1-day shift) train/test splits."""
    raw = yf.download(TICKERS, start="2015-01-01", end="2023-07-01", progress=False)
    prices = raw["Close"][TICKERS].dropna()
    returns = prices.pct_change().dropna()

    # Shift features by 1 day so state s_t only sees returns through t-1.
    roll_mean = returns.rolling(20).mean() * 20.0
    roll_vol = returns.rolling(20).std() * np.sqrt(20.0)
    mkt_feats = pd.concat([roll_mean, roll_vol], axis=1).shift(1).dropna()
    returns = returns.loc[mkt_feats.index]

    train_mask = mkt_feats.index < "2022-01-01"
    test_mask = mkt_feats.index >= "2022-01-01"

    feat_mean = mkt_feats.loc[train_mask].mean()
    feat_std = mkt_feats.loc[train_mask].std() + 1e-8
    mkt_norm = ((mkt_feats - feat_mean) / feat_std).values.astype(np.float32)

    ret_train = torch.tensor(returns.loc[train_mask].values, dtype=torch.float32, device=device)
    feat_train = torch.tensor(mkt_norm[train_mask], dtype=torch.float32, device=device)
    ret_test = torch.tensor(returns.loc[test_mask].values, dtype=torch.float32, device=device)
    feat_test = torch.tensor(mkt_norm[test_mask], dtype=torch.float32, device=device)
    test_dates = returns.loc[test_mask].index
    return ret_train, feat_train, ret_test, feat_test, test_dates


class GPUPortfolioEnv:
    def __init__(self, rets, feats, tc_bps=10.0):
        self.rets = rets
        self.feats = feats
        self.device = rets.device
        self.T = rets.shape[0]
        self.n_assets = rets.shape[1]
        self.tc = tc_bps / 10000.0
        self.state_dim = feats.shape[1] + self.n_assets + 2
        self.action_dim = self.n_assets

    def build_state(self, t_idx, weights, log_wealth, drawdown):
        f = self.feats[t_idx]
        return torch.cat([f, weights, log_wealth.unsqueeze(-1), drawdown.unsqueeze(-1)], dim=-1)

    @torch.no_grad()
    def step(self, t_idx, weights, log_wealth, peak_wealth, target_weights):
        """target_weights must already lie on the simplex (sum to 1, >= 0)."""
        turnover = torch.abs(target_weights - weights).sum(dim=-1)
        asset_r = self.rets[t_idx]
        port_r = (target_weights * asset_r).sum(dim=-1) - self.tc * turnover
        port_r = port_r.clamp(-0.99, 1.0)

        new_log_wealth = log_wealth + torch.log1p(port_r)
        new_peak = torch.maximum(peak_wealth, new_log_wealth)
        new_dd = new_log_wealth - new_peak
        next_t = (t_idx + 1).clamp(max=self.T - 1)
        next_s = self.build_state(next_t, target_weights, new_log_wealth, new_dd)
        return next_s, target_weights, new_log_wealth, new_peak, new_dd, port_r

    @torch.no_grad()
    def collect_rollouts(self, n_traj=2048, horizon=60):
        """Reward-free exploration over 2015-2021 storing true simplex weights."""
        device = self.device
        states = torch.zeros((n_traj, horizon + 1, self.state_dim), device=device)
        actions = torch.zeros((n_traj, horizon, self.action_dim), device=device)

        t_idx = torch.randint(0, self.T - horizon - 2, (n_traj,), device=device)
        w = torch.full((n_traj, self.n_assets), 1.0 / self.n_assets, device=device)
        lw = torch.zeros(n_traj, device=device)
        pk = torch.zeros(n_traj, device=device)
        dd = torch.zeros(n_traj, device=device)

        states[:, 0] = self.build_state(t_idx, w, lw, dd)
        logits = torch.randn((n_traj, self.n_assets), device=device) * 1.8
        for h in range(horizon):
            if h % 5 == 0:
                logits = torch.randn((n_traj, self.n_assets), device=device) * 1.8
            else:
                logits = (0.85 * logits + 0.40 * torch.randn_like(logits)).clamp(-3.5, 3.5)
            w_act = F.softmax(logits, dim=-1)
            actions[:, h] = w_act
            s_next, w, lw, pk, dd, _ = self.step(t_idx + h, w, lw, pk, w_act)
            states[:, h + 1] = s_next
        return states, actions


@torch.no_grad()
def measure_action_sensitivity(model, s_probe):
    """Compare 100% SPY [1,0,0,0,0] vs. 100% SHY [0,0,0,0,1] in latent space."""
    device = s_probe.device
    z = model.enc(s_probe)
    B = s_probe.shape[0]
    w_spy = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device).expand(B, -1)
    w_shy = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], device=device).expand(B, -1)
    s1 = F.mse_loss(model.step1(z, w_spy), model.step1(z, w_shy)).item()
    s2 = F.mse_loss(model.step2(z, w_spy), model.step2(z, w_shy)).item()
    return s1, s2


def compute_portfolio_metrics(daily_rets):
    daily_rets = np.array(daily_rets)
    equity = np.cumprod(1.0 + daily_rets)
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity / peak) - 1.0).min() * 100.0
    cum_ret = (equity[-1] - 1.0) * 100.0
    sharpe = (daily_rets.mean() / (daily_rets.std() + 1e-8)) * np.sqrt(252)
    return max_dd, cum_ret, sharpe, equity


@torch.no_grad()
def eval_jepa_rate_shock(test_env, model, hierarchical=True):
    device = test_env.device
    n_assets = test_env.n_assets
    w = torch.full((1, n_assets), 1.0 / n_assets, device=device)
    lw = torch.zeros(1, device=device)
    pk = torch.zeros(1, device=device)
    dd = torch.zeros(1, device=device)
    s_curr = test_env.build_state(torch.tensor([0], device=device), w, lw, dd)

    def make_goal_state(t_i, cur_lw):
        g_w = torch.tensor([[0.05, 0.05, 0.20, 0.20, 0.50]], device=device)
        g_lw = cur_lw + 0.05
        g_dd = torch.zeros_like(cur_lw)
        return test_env.build_state(torch.tensor([t_i], device=device), g_w, g_lw, g_dd)

    daily_rets = []
    z_subgoal = model.enc(make_goal_state(0, lw))

    for t in range(test_env.T - 1):
        z_curr = model.enc(s_curr)
        z_goal = model.enc(make_goal_state(t, lw))

        if hierarchical:
            if t % 5 == 0:
                def cost_l2(w_seqs):
                    P, H, _ = w_seqs.shape
                    z = z_curr.expand(P, -1)
                    for h in range(H):
                        z = model.step2(z, w_seqs[:, h])
                    return ((z - z_goal) ** 2).mean(dim=-1)

                best_macro = batched_cem_simplex(cost_l2, horizon=4, act_dim=n_assets, device=device, pop_size=256, n_iters=4)
                z_subgoal = model.step2(z_curr, best_macro[0:1])

            def cost_l1(w_seqs):
                P, K, _ = w_seqs.shape
                z = z_curr.expand(P, -1)
                for k_s in range(K):
                    z = model.step1(z, w_seqs[:, k_s])
                return 0.50 * ((z - z_subgoal) ** 2).mean(dim=-1) + 0.50 * ((z - z_goal) ** 2).mean(dim=-1)

            best_w = batched_cem_simplex(cost_l1, horizon=model.k, act_dim=n_assets, device=device, pop_size=192, n_iters=4)
        else:
            def cost_flat(w_seqs):
                P, K, _ = w_seqs.shape
                z = z_curr.expand(P, -1)
                for k_s in range(K):
                    z = model.step1(z, w_seqs[:, k_s])
                return ((z - z_goal) ** 2).mean(dim=-1)

            best_w = batched_cem_simplex(cost_flat, horizon=2, act_dim=n_assets, device=device, pop_size=256, n_iters=4)

        s_curr, w, lw, pk, dd, port_r = test_env.step(torch.tensor([t], device=device), w, lw, pk, best_w[0:1])
        daily_rets.append(port_r.item())

    return compute_portfolio_metrics(daily_rets)


def train_and_eval_ppo_finance(train_env, test_env, num_envs=256, rollout_len=60, updates=40):
    device = train_env.device
    n_assets = train_env.n_assets
    ppo = PortfolioPPO(train_env.state_dim, train_env.action_dim).to(device)
    opt = torch.optim.Adam(ppo.parameters(), lr=3e-3)

    for _ in range(updates):
        t_idx = torch.randint(0, train_env.T - rollout_len - 2, (num_envs,), device=device)
        w = torch.full((num_envs, n_assets), 1.0 / n_assets, device=device)
        lw = torch.zeros(num_envs, device=device)
        pk = torch.zeros(num_envs, device=device)
        dd = torch.zeros(num_envs, device=device)
        s = train_env.build_state(t_idx, w, lw, dd)

        obs_b, act_b, logp_b, rew_b, val_b = [], [], [], [], []
        for h in range(rollout_len):
            with torch.no_grad():
                act, logp, _, val = ppo.get_action_and_value(s)
            w_act = F.softmax(act, dim=-1)
            s_next, w, lw, pk, dd, port_r = train_env.step(t_idx + h, w, lw, pk, w_act)
            rew = port_r * 100.0 + 0.5 * dd
            obs_b.append(s)
            act_b.append(act)
            logp_b.append(logp)
            rew_b.append(rew)
            val_b.append(val)
            s = s_next

        returns_t = torch.zeros((rollout_len, num_envs), device=device)
        R = torch.zeros(num_envs, device=device)
        for t in reversed(range(rollout_len)):
            R = rew_b[t] + 0.99 * R
            returns_t[t] = R

        b_obs = torch.stack(obs_b).reshape(-1, train_env.state_dim)
        b_act = torch.stack(act_b).reshape(-1, train_env.action_dim)
        b_logp = torch.stack(logp_b).reshape(-1)
        b_ret = returns_t.reshape(-1)
        b_adv = b_ret - torch.stack(val_b).reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(4):
            _, new_logp, ent, new_val = ppo.get_action_and_value(b_obs, b_act)
            ratio = torch.exp(new_logp - b_logp)
            loss = (
                -torch.min(ratio * b_adv, torch.clamp(ratio, 0.8, 1.2) * b_adv).mean()
                + 0.5 * F.mse_loss(new_val, b_ret)
                - 0.01 * ent.mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        w = torch.full((1, n_assets), 1.0 / n_assets, device=device)
        lw = torch.zeros(1, device=device)
        pk = torch.zeros(1, device=device)
        dd = torch.zeros(1, device=device)
        s = test_env.build_state(torch.tensor([0], device=device), w, lw, dd)
        daily_rets = []
        for t in range(test_env.T - 1):
            w_act = F.softmax(ppo.actor_mean(s), dim=-1)
            s, w, lw, pk, dd, port_r = test_env.step(torch.tensor([t], device=device), w, lw, pk, w_act)
            daily_rets.append(port_r.item())

    return compute_portfolio_metrics(daily_rets)


def run_experiment():
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Leak-Free Finance Regime-Shift Study (2022 Rate Shock) on: {device}")

    ret_train, feat_train, ret_test, feat_test, test_dates = load_rate_shock_data(device)
    print(f"Train days (2015-2021): {len(ret_train)} | OOD Rate-Shock Test days (2022-2023): {len(ret_test)}")

    train_env = GPUPortfolioEnv(ret_train, feat_train)
    test_env = GPUPortfolioEnv(ret_test, feat_test)

    conditions = [
        "H-JEPA + SIGReg + InvDyn (Full)",
        "H-JEPA + SIGReg (No InvDyn)",
        "Flat 1-Level JEPA + InvDyn",
        "PPO Actor-Critic (2015-21 RL)",
    ]
    mdd_res = {c: [] for c in conditions}
    ret_res = {c: [] for c in conditions}
    sharpe_res = {c: [] for c in conditions}
    sens_logs = {"With InvDyn": [], "No InvDyn": []}
    eq_curves = {}

    r_test_np = ret_test.cpu().numpy()[:-1]
    eq_6040 = np.cumprod(1.0 + 0.60 * r_test_np[:, 0] + 0.40 * r_test_np[:, 1])

    for seed in range(N_SEEDS):
        torch.manual_seed(200 + seed)
        np.random.seed(200 + seed)

        states, actions = train_env.collect_rollouts(n_traj=2048, horizon=60)
        s_probe = states[:256, 0]

        model_full = train_hjepa(
            states, actions, use_inv_head=True, macro_stride=5, rollout_h=4, epochs=250, batch_size=256, simplex_actions=True
        )
        _, s2_f = measure_action_sensitivity(model_full, s_probe)
        sens_logs["With InvDyn"].append(s2_f)

        mdd_f, ret_f, sh_f, eq_f = eval_jepa_rate_shock(test_env, model_full, hierarchical=True)
        mdd_fl, ret_fl, sh_fl, eq_fl = eval_jepa_rate_shock(test_env, model_full, hierarchical=False)

        model_noinv = train_hjepa(
            states, actions, use_inv_head=False, macro_stride=5, rollout_h=4, epochs=250, batch_size=256, simplex_actions=True
        )
        _, s2_n = measure_action_sensitivity(model_noinv, s_probe)
        sens_logs["No InvDyn"].append(s2_n)
        mdd_n, ret_n, sh_n, eq_n = eval_jepa_rate_shock(test_env, model_noinv, hierarchical=True)

        mdd_p, ret_p, sh_p, eq_p = train_and_eval_ppo_finance(train_env, test_env)

        for name, mdd, cr, sh in [
            ("H-JEPA + SIGReg + InvDyn (Full)", mdd_f, ret_f, sh_f),
            ("H-JEPA + SIGReg (No InvDyn)", mdd_n, ret_n, sh_n),
            ("Flat 1-Level JEPA + InvDyn", mdd_fl, ret_fl, sh_fl),
            ("PPO Actor-Critic (2015-21 RL)", mdd_p, ret_p, sh_p),
        ]:
            mdd_res[name].append(mdd)
            ret_res[name].append(cr)
            sharpe_res[name].append(sh)

        if seed == 0:
            eq_curves = {
                "H-JEPA + SIGReg + InvDyn (Full)": eq_f,
                "H-JEPA + SIGReg (No InvDyn)": eq_n,
                "Flat 1-Level JEPA + InvDyn": eq_fl,
                "PPO Actor-Critic (2015-21 RL)": eq_p,
            }

        print(
            f"Seed {seed + 1}/{N_SEEDS} | 2022 MaxDD -> Full: {mdd_f:6.2f}% | "
            f"No-Inv: {mdd_n:6.2f}% (Sens:{s2_n:.4f}) | "
            f"Flat: {mdd_fl:6.2f}% | PPO: {mdd_p:6.2f}% | InvDyn Sens:{s2_f:.4f}"
        )

    groups_mdd = [np.array(mdd_res[c]) for c in conditions]
    kw_mdd, p_kw_mdd = kruskal_wallis(groups_mdd)

    print("\n" + "=" * 92)
    print("LEAK-FREE 2022 RATE-SHOCK ZERO-SHOT SUMMARY (N = 8 SEEDS | TRAINED 2015-2021 ONLY)")
    print("-" * 92)
    print(f"{'Condition':<35} | {'Max Drawdown %':<16} | {'Cum Return %':<15} | {'Sharpe'}")
    print("-" * 92)
    for c in conditions:
        print(
            f"{c:<35} | {np.mean(mdd_res[c]):6.2f} ± {np.std(mdd_res[c], ddof=1):5.2f}% | "
            f"{np.mean(ret_res[c]):+6.2f} ± {np.std(ret_res[c], ddof=1):5.2f}% | "
            f"{np.mean(sharpe_res[c]):+5.2f}"
        )
    print("-" * 92)
    print(f"  Kruskal-Wallis Omnibus (Max Drawdown): H = {kw_mdd:.3f}, p = {p_kw_mdd:.3e}")
    print(f"  Action Sensitivity (100% SPY vs SHY) -> With InvDyn: {np.mean(sens_logs['With InvDyn']):.4f} | No InvDyn: {np.mean(sens_logs['No InvDyn']):.4f}")
    print("=" * 92)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    colors = ["#2980b9", "#95a5a6", "#8e44ad", "#e67e22"]
    short_labels = ["H-JEPA\n+InvDyn", "H-JEPA\nNo-InvDyn", "Flat JEPA", "PPO RL\n(2015-21)"]

    for (name, eq), col in zip(eq_curves.items(), colors):
        axes[0].plot(test_dates[:-1], eq, label=name.split(" (")[0], color=col, lw=2.0)
    axes[0].plot(test_dates[:-1], eq_6040, label="60/40 (SPY/TLT)", color="black", linestyle="--", lw=1.5, alpha=0.7)
    axes[0].set_title("Leak-Free 2022 Rate-Shock Equity Curves (Seed 0)", fontsize=10.5)
    axes[0].set_ylabel("Portfolio Value ($1 Start)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    bp1 = axes[1].boxplot(groups_mdd, patch_artist=True, widths=0.45, medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp1["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    for idx, g in enumerate(groups_mdd, 1):
        axes[1].scatter(np.full_like(g, idx) + np.random.normal(0, 0.04, len(g)), g, color="black", s=24, zorder=4)
    axes[1].set_xticklabels(short_labels, fontsize=9)
    axes[1].set_ylabel("Max Drawdown (%) — Higher (Closer to 0) is Better")
    axes[1].set_title(f"2022 Max Drawdown (KW p={p_kw_mdd:.1e})", fontsize=10.5)
    axes[1].grid(True, axis="y", alpha=0.25)

    groups_ret = [np.array(ret_res[c]) for c in conditions]
    bp2 = axes[2].boxplot(groups_ret, patch_artist=True, widths=0.45, medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp2["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    for idx, g in enumerate(groups_ret, 1):
        axes[2].scatter(np.full_like(g, idx) + np.random.normal(0, 0.04, len(g)), g, color="black", s=24, zorder=4)
    axes[2].set_xticklabels(short_labels, fontsize=9)
    axes[2].set_ylabel("Cumulative Return (%) — Higher is Better")
    axes[2].set_title("2022–2023 Out-of-Distribution Return", fontsize=10.5)
    axes[2].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
