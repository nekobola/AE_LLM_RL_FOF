import torch
import numpy as np
import yaml
import pickle
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '.')

from src.ppo.networks import ActorCritic
from src.env.action_mapper import ActionMapper
from src.env.mdp_environment import MDPEnvironment
from src.models.regime_autoencoder import RegimeAutoEncoder
from src.compute.dual_track_engine import DualTrackEngine

# Load config
with open('config.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

ppo_cfg = config.get('ppo', {})
paths_cfg = config.get('paths', {})

# Load checkpoint
ckpt = torch.load('checkpoints/actor_critic.pth', map_location='cpu')

# Initialize model
ac = ActorCritic(
    state_dim=ppo_cfg.get('state_dim', 10),
    action_dim=ppo_cfg.get('action_dim', 2),
    hidden_dim=64,
)
ac.load_state_dict(ckpt['ac'])
ac.eval()

# Action mapper
am_cfg = config.get('action_mapper', {})
action_mapper = ActionMapper(
    alpha_min=am_cfg.get('alpha_min', -0.5),
    alpha_max=am_cfg.get('alpha_max', 0.5),
    tau_delta_range=am_cfg.get('tau_delta_range', 0.1),
    alpha_bias=am_cfg.get('alpha_bias', 0.0),
)

# Load features
features_df = pd.read_parquet(
    Path(paths_cfg.get('data_processed', 'data/processed')) / 'features_master.parquet'
)

# Load AE model
model_cfg = config.get('model', {}).get('regime_autoencoder', {})
ae_model = RegimeAutoEncoder(
    input_dim=features_df.shape[1],
    latent_dim=model_cfg.get('latent_dim', 6),
    hidden_dim=model_cfg.get('hidden_dim', 16),
)
ae_ckpt = torch.load(
    Path(paths_cfg.get('checkpoints', 'checkpoints')) / 'ae_weights.pth',
    map_location='cpu', weights_only=True
)
ae_model.load_state_dict(ae_ckpt.get('model_state', ae_ckpt))
ae_model.eval()

# Compute AE errors
X = features_df.values
ae_errors = []
for i in range(len(X)):
    x = torch.from_numpy(X[i:i+1]).float()
    with torch.no_grad():
        recon = ae_model(x)
        error = float(torch.sum((x - recon) ** 2).item())
    ae_errors.append(error)
ae_errors = np.array(ae_errors)

print('=' * 60)
print('  Full Environment Rollout: Alpha Distribution Test')
print('=' * 60)
print()

env = MDPEnvironment(config)
state, _ = env.reset()
state = np.array(state, dtype=np.float32)

alphas = []
taus = []
rewards = []
r_ports = []
r_normals = []

# Get asset return columns
asset_codes = config.get('data_pipeline', {}).get('asset_codes', [
    '000300.SH', '000852.SH', 'CBA02701.CS', 'AU9999.SGE', 'NH0100.NHF'
])
weekly_return_cols = [f'{code}__weekly_return' for code in asset_codes]
available_cols = [c for c in weekly_return_cols if c in features_df.columns]

dual_engine = DualTrackEngine(config)

# Run through all historical data
for i in range(min(len(features_df), 1200)):
    ae_error = ae_errors[i]
    vol = 0.15
    llm_macro, llm_sentiment, llm_risk = 50.0, 50.0, 50.0

    if available_cols and i < len(features_df):
        asset_returns = features_df.iloc[i][available_cols].fillna(0.0).values
    else:
        asset_returns = np.zeros(5)

    lookback = min(30, i)
    if available_cols:
        ret_5d = features_df[available_cols].fillna(0.0).values[max(0,i-lookback):i+1].T
    else:
        ret_5d = np.zeros((5, 1))

    if ret_5d.shape[1] >= 5:
        w_normal, w_event = dual_engine.compute(ret_5d, llm_macro, llm_sentiment, llm_risk)
    else:
        w_normal = np.array([0.2]*5)
        w_event = np.array([0.2]*5)

    returns_window = None
    if i >= 1 and available_cols:
        ret_prev = features_df.iloc[i-1][available_cols].fillna(0.0).values.tolist()
        ret_curr = features_df.iloc[i][available_cols].fillna(0.0).values.tolist()
        returns_window = [ret_prev, ret_curr]

    live_data = {
        'ae_error': float(ae_error),
        'vol_mkt_20d': vol,
        'llm_macro': llm_macro,
        'llm_sentiment': llm_sentiment,
        'llm_risk': llm_risk,
        'asset_returns_t': asset_returns.tolist() if len(asset_returns) == 5 else [0.0]*5,
        'w_normal_t': w_normal.tolist(),
        'w_event_t': w_event.tolist(),
        'returns_window_5d': returns_window,
    }

    env.inject_live_data(live_data)

    # Get action from trained policy
    with torch.no_grad():
        state_t = torch.from_numpy(state).unsqueeze(0).float()
        mu_t, _ = ac.actor(state_t)
        action_t = torch.tanh(mu_t)

    action_np = action_t.numpy().squeeze().astype(np.float32)

    next_state, reward, terminated, truncated, info = env.step(action_np)

    alphas.append(info['alpha'])
    taus.append(info['tau'])
    rewards.append(reward)
    r_ports.append(info['r_port'])
    r_normals.append(info['r_normal'])

    state = np.array(next_state, dtype=np.float32)

    if terminated or truncated:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

alphas = np.array(alphas)
taus = np.array(taus)
rewards = np.array(rewards)
r_ports = np.array(r_ports)
r_normals = np.array(r_normals)

# Results
print('=== Alpha Distribution (100k steps trained) ===')
print('  Mean:   %.4f' % np.mean(alphas))
print('  Std:    %.4f' % np.std(alphas))
print('  Min:    %.4f' % np.min(alphas))
print('  Max:    %.4f' % np.max(alphas))
print('  Median: %.4f' % np.median(alphas))
print()

# Histogram
print('Alpha histogram:')
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(alphas, bins=bins)
for i in range(len(hist)):
    bar = '#' * int(hist[i] / max(1, len(alphas)) * 100)
    print('  [%.1f, %.1f): %5d (%5.1f%%) %s' % (bins[i], bins[i+1], hist[i], hist[i]/len(alphas)*100, bar))
print()

print('=== Tau Distribution ===')
print('  Mean:   %.2f' % np.mean(taus))
print('  Std:    %.2f' % np.std(taus))
print('  Min:    %.2f' % np.min(taus))
print('  Max:    %.2f' % np.max(taus))
print()

print('=== Reward Distribution ===')
print('  Mean:   %.4f' % np.mean(rewards))
print('  Std:    %.4f' % np.std(rewards))
print('  Min:    %.4f' % np.min(rewards))
print('  Max:    %.4f' % np.max(rewards))
print()

# Check regime awareness
ae_errors_subset = ae_errors[:len(alphas)]
high_error_mask = ae_errors_subset > 20.0
low_error_mask = ae_errors_subset <= 20.0

if np.sum(high_error_mask) > 0 and np.sum(low_error_mask) > 0:
    print('=== Regime Awareness ===')
    print('  High AE error (>=20): alpha mean=%.4f, n=%d' % (np.mean(alphas[high_error_mask]), np.sum(high_error_mask)))
    print('  Low AE error (<20):   alpha mean=%.4f, n=%d' % (np.mean(alphas[low_error_mask]), np.sum(low_error_mask)))
    diff = np.mean(alphas[low_error_mask]) - np.mean(alphas[high_error_mask])
    print('  Difference (low - high): %.4f' % diff)
    if abs(diff) > 0.05:
        print('  >> Model shows regime-aware behavior!')
    else:
        print('  >> Model does NOT differentiate between regimes')

print()
print('=== Return Comparison ===')
print('  Portfolio mean return:   %.6f' % np.mean(r_ports))
print('  NormalTrack mean return: %.6f' % np.mean(r_normals))
print('  Alpha (excess):          %.6f' % (np.mean(r_ports) - np.mean(r_normals)))
