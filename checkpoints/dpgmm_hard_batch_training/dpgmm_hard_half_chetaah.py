import os

# For headless MuJoCo rendering on a remote GPU server
os.environ["MUJOCO_GL"] = "egl"


from dataclasses import dataclass, asdict
import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.utils.parametrizations as param
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
import pandas as pd
from pathlib import Path
import json
import mujoco
import random
import copy
import time
from scipy.special import gammaln
from sklearn.metrics import adjusted_rand_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, silhouette_score
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R



@dataclass
class LDRConfig:
    # SAC and training settings (Xu: configs/agents/hopper/ldr.yaml)
    total_timesteps: int = 1_500_000
    learning_starts: int = 0             # Xu: train_start_iterations = 0, policy acts from the start
    train_freq: int = 2048               # env steps collected per optimize() call
    gradient_steps: int = 2048           # gradient updates per optimize() call (1 per env step)
    batch_size: int = 256
    eval_interval: int = 10_000
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    reward_scale: float = 0.008
    target_update_interval: int = 1
    hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    seed: int = 1012

    # LDR-specific settings
    n_past: int = 50
    latent_dim: int = 3                  # Xu uses 5 only for Ant
    vae_loss_weight: float = 5.0
    contrastive_loss_weight: float = 1.0
    infonce_temp: float = 1.0
    contrastive_start_iterations: int = 200_000   # encoder/decoder warm-up (RL + VAE only) before clustering starts
    latent_dropout_prob: float = 0.1
    min_positives: int = 1

    # KL annealing (Xu's kl_schedule): beta goes 0 -> beta_max linearly
    # over the first beta_anneal_fraction of training, then stays at beta_max
    beta_max: float = 0.1
    beta_anneal_fraction: float = 0.5
    encoder_reg_mode: str = "none" #specify "none", "l2", or "spectral" or "orthogonal" for encoder regularization

    # DP-GMM clustering of OUTCOME features (teacher-decoder predictions on shared probes)
    ema_rho: float = 0.995               # teacher EMA momentum (encoder and decoder)
    dp_refresh_interval: int = 100_000   # env steps between DP-GMM refreshes
    dp_num_points: int = 1500            # replay windows clustered per refresh
    dp_alpha: float = 1.0                # CRP concentration
    dp_kappa0: float = 0.01              # NIW kappa_0
    dp_nu0: float = None                 # NIW nu_0 (None -> d + 2)
    dp_cluster_std: float = 0.3          # prior within-cluster std, in units of the global outcome std
    dp_burn: int = 30                    # Gibbs burn-in sweeps
    dp_samples: int = 20                 # retained Gibbs samples
    dp_thin: int = 2                     # sweeps between retained samples
    num_probes: int = 64                 # M shared (s, a) reference pairs, resampled at each refresh
    outcome_pca_var: float = 0.99        # keep the fewest PCA dims of the outcome vector reaching this variance ...
    outcome_pca_dim: int = 8             # ... but never more than this many
    outcome_chunk: int = 256             # histories per teacher-decoder pass when building outcome vectors

    # True : HARD pseudo-labels (one representative partition; same cluster -> positive pair, as in K-means)
    # False: SOFT targets (posterior co-clustering probabilities S_ij as positive-pair weights)
    dp_hard_labels: bool = True
    dp_min_positive_mass: float = 0.5    # soft only: anchor valid if sum_{l!=i} S_il exceeds this


ENV_PRESETS = {
    "Hopper-v5":      dict(latent_dim=3, reward_scale=0.008, beta_max=0.10, learning_rate=3e-4, decoder_hidden_dim=256),
    "Walker2d-v5":    dict(latent_dim=3, reward_scale=0.004, beta_max=0.07, learning_rate=3e-4, decoder_hidden_dim=256),
    "HalfCheetah-v5": dict(latent_dim=3, reward_scale=0.004, beta_max=0.07, learning_rate=3e-4, decoder_hidden_dim=256),
    "Ant-v5":         dict(latent_dim=5, reward_scale=0.006, beta_max=0.04, learning_rate=1e-4, decoder_hidden_dim=512),
}


def applyLipschitzRegularization(module, reg_mode):
    """Recursively apply spectral/orthogonal parametrization to Linear and Conv1d children (Xu)."""
    if reg_mode not in ("spectral", "orthogonal", "none"):
        raise ValueError(f"Invalid reg_mode '{reg_mode}'")
    if reg_mode == "none":
        return module
    for child in module.children():
        if isinstance(child, (nn.Linear, nn.Conv1d)):
            if reg_mode == "spectral":
                param.spectral_norm(child, name="weight")
            else:
                param.orthogonal(child, name="weight")
        else:
            applyLipschitzRegularization(child, reg_mode)
    return module


class HistoryQueue:
    def __init__(self, n_envs, n_past, obs_dim, action_dim):
        self.n_envs, self.H, self.obs_dim, self.action_dim = n_envs, n_past, obs_dim, action_dim
        self.data = np.zeros((n_envs, n_past, obs_dim + action_dim), dtype=np.float32)
    
    def get(self):
        return self.data.copy()
    
    def append(self, obs, actions):
        # shifting all data except the last 1 row. then add the new obs and actions to the first row
        self.data[:,1:] = self.data[:,:-1].copy()
        self.data[:,0,:self.obs_dim] = obs
        self.data[:,0,self.obs_dim:] = actions
    
    # def reset_done(self, dones):
    #     self.data[np.asarray(dones, dtype=bool)] = 0

    def reset_done(self, dones, initial_obs):
        dones = np.asarray(dones, dtype=bool)
        self.data[dones] = 0
        if initial_obs is not None:
            self.data[dones, :, :self.obs_dim] = initial_obs[dones][:, None, :]



# Need to send latent_dim from the LDRAgent

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=256, latent_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim > 0:
            self.latent_norm = nn.LayerNorm(latent_dim)

        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Gaussian distribution parameters
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, obs):
        # obs = [env_obs, z]; normalize z before the MLP
        if self.latent_dim > 0:
            s, z = obs[..., :-self.latent_dim], obs[..., -self.latent_dim:]
            obs = torch.cat([s, self.latent_norm(z)], dim=-1)

        features = self.net(obs)
        mean = self.mu(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  
        std = log_std.exp()

        return mean, std


    # Sample actions and compute log probabilities
    def actionLogProb(self, obs):

        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        u = dist.rsample() # reparameterization trick for backpropagation
        action = torch.tanh(u) # action between -1 and 1

        # Calculting log \pi(a|S)
        log_prob = dist.log_prob(u)
        log_prob -= 2 * (torch.log(torch.tensor(2.0, device=u.device)) - u - nn.functional.softplus(-2 * u))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def predict(self, obs, deterministic=True):
        """Return normalized actions for inference."""

        mean, std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean)

        dist = Normal(mean, std)
        return torch.tanh(dist.sample())


class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=256, latent_dim=0):
        super().__init__()

        input_dim = obs_dim + action_dim
        self.latent_dim = latent_dim

        # First Q network
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Second Q network
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # One LayerNorm per Q head (Xu deep-copies the extractor for each head)
        if latent_dim > 0:
            self.norm1 = nn.LayerNorm(latent_dim)
            self.norm2 = nn.LayerNorm(latent_dim)

    def _input(self, obs, action, norm):
        if self.latent_dim > 0:
            s, z = obs[..., :-self.latent_dim], obs[..., -self.latent_dim:]
            obs = torch.cat([s, norm(z)], dim=-1)
        return torch.cat([obs, action], dim=-1)

    def forward(self, obs, action):
        if self.latent_dim > 0:
            q1 = self.q1(self._input(obs, action, self.norm1))
            q2 = self.q2(self._input(obs, action, self.norm2))
        else:
            x = torch.cat([obs, action], dim=-1)
            q1, q2 = self.q1(x), self.q2(x)

        return q1, q2



class ReplayBuffer:

    def __init__(self, buffer_size, num_envs, obs_dim, action_dim, device, n_past):

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Capacity measured in vectorized steps
        self.capacity = max(buffer_size // num_envs, 1 )

        # Allocate storage
        self.obs = np.zeros((self.capacity, num_envs, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((self.capacity, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, num_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, num_envs), dtype=np.float32)
        self.timeouts = np.zeros_like(self.dones)

        # LDR specific
        self.histories = np.zeros((self.capacity, num_envs, n_past, obs_dim + action_dim), dtype=np.float32)
        self.dynamics_ids = np.zeros((self.capacity, num_envs), dtype=np.int64)

        self.pos = 0
        self.full = False

    def push(self, obs, next_obs, actions, rewards, dones, timeouts, history, dynamics_ids):

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.timeouts[self.pos] = timeouts
        self.histories[self.pos] = history
        self.dynamics_ids[self.pos] = dynamics_ids

        # Circular buffer
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.full = True

    # bookkeeping, length of stored transition
    def __len__(self):
        steps = self.capacity if self.full else self.pos
        return steps * self.num_envs

    def sample(self, batch_size):

        size = self.capacity if self.full else self.pos

        if size == 0:
            raise ValueError("Cannot sample an empty buffer")

        # Random time & environment indices
        batch_indices = np.random.randint(0, size, size=batch_size)
        env_indices = np.random.randint(0, self.num_envs, size=batch_size)

        obs = self.obs[batch_indices, env_indices]
        next_obs = self.next_obs[batch_indices, env_indices]
        actions = self.actions[batch_indices, env_indices]
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, 1) #reshapes into a column vector for each transition
        dones = self.dones[batch_indices, env_indices].reshape(-1, 1)
        timeouts = self.timeouts[batch_indices, env_indices].reshape(-1, 1)
        history= self.histories[batch_indices, env_indices]
        ids = self.dynamics_ids[batch_indices, env_indices]

        # Only true terminations stop bootstrapping
        dones = dones * (1.0 - timeouts)

        def to_tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        return {
            "obs": to_tensor(obs),
            "next_obs": to_tensor(next_obs),
            "actions": to_tensor(actions),
            "rewards": to_tensor(rewards),
            "dones": to_tensor(dones),
            "history": to_tensor(history),
            "ids": torch.as_tensor(ids, dtype=torch.long, device=self.device)
        }

    def sample_histories(self, n):
        """Distinct (time, env) windows for DP-GMM clustering."""
        size = self.capacity if self.full else self.pos
        flat = np.random.choice(size * self.num_envs, size=min(n, size * self.num_envs), replace=False)
        t, e = np.divmod(flat, self.num_envs)
        return self.histories[t, e], self.dynamics_ids[t, e]



class DynamicsEncoder(nn.Module):

    def __init__(
            self, 
            obs_dim, 
            action_dim, 
            H, 
            latent_dim, 
            embed_dim=32, 
            kernel_sizes=(8, 5, 5), 
            strides=(4, 1, 1), 
            log_std_min=-20.0, 
            log_std_max=2.0,
            reg_mode="none"
            ):
        super().__init__()
        input_dim = obs_dim + action_dim

        # Per-timestep (obs, action) embedding, shared across time; no ReLU on the output
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # TCN: Conv1d + ReLU blocks over the time axis
        tcn_layers = []
        for k, s in zip(kernel_sizes, strides):
            tcn_layers += [nn.Conv1d(embed_dim, embed_dim, kernel_size=k, stride=s), nn.ReLU()]
        self.tcn = nn.Sequential(*tcn_layers)

        # Flattened TCN output size (96 for H=50 with the default kernels/strides)
        with torch.no_grad():
            dummy = torch.zeros(1, H, input_dim)
            flat_size = self.tcn(self.embedding(dummy).permute(0, 2, 1)).reshape(1, -1).size(1)

        self.mean = nn.Linear(flat_size, latent_dim)
        self.log_std = nn.Linear(flat_size, latent_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # lipchitz regularization
        self.embedding = applyLipschitzRegularization(self.embedding, reg_mode)
        self.tcn = applyLipschitzRegularization(self.tcn, reg_mode)

    def forward(self, history):
        # history: (B, H, obs+act), newest first -> flip to oldest first
        x = torch.flip(history, dims=[1])
        x = self.embedding(x).permute(0, 2, 1)       # (B, embed_dim, H)
        x = self.tcn(x).reshape(x.size(0), -1)       # (B, flat_size)

        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return Normal(self.mean(x), log_std.exp())




class RunningMeanStd(nn.Module):
    """Running mean/var of the target delta (buffers, not trained)."""

    def __init__(self, shape, epsilon=1e-5, clip_limit=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))
        self.epsilon = epsilon
        self.clip_limit = clip_limit

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        M2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(M2 / tot_count)
        self.count.copy_(tot_count)

    def normalize(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.epsilon), -self.clip_limit, self.clip_limit)

    def denormalize(self, x):
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


class DynamicsDecoder(nn.Module):

    def __init__(self, obs_dim, action_dim, latent_dim, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim

        # LayerNorm on z before the MLP (Xu's LatentMLPFeatureExtractor backbone)
        self.latent_norm = nn.LayerNorm(latent_dim)

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        # Predicts the NORMALIZED change in observation
        self.mean = nn.Linear(hidden, obs_dim)
        self.log_std = nn.Linear(hidden, obs_dim)

        self.target_scaler = RunningMeanStd(shape=(obs_dim,))

    def forward(self, obs, action, z):
        x = self.net(torch.cat((obs, action, self.latent_norm(z)), dim=-1))
        return Normal(self.mean(x), self.log_std(x).clamp(-20, 2).exp())

    def logProb(self, obs, next_obs, action, z):
        """Log prob of the observed transition, with target normalization (as in Xu)."""
        delta_obs = next_obs - obs
        if self.training:
            self.target_scaler.update(delta_obs.detach())
        return self(obs, action, z).log_prob(self.target_scaler.normalize(delta_obs))

    @torch.no_grad()
    def predictNextObs(self, obs, action, z):
        """Denormalized next-obs prediction (for visualization)."""
        delta = self.target_scaler.denormalize(self(obs, action, z).mean)
        return obs + delta



def multi_positive_infonce(z, ids, temperature):
    z = F.normalize(z, dim=-1) #projecting onto unit sphere

    logits = z @ z.T / temperature  # Matrix of pairwise similarities

    self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device) # Self similarities, diagonal of the logits matrix
    positive = (ids[:, None] == ids[None, :]) & ~self_mask
    counts = positive.sum(dim=1)
    valid = counts > 0

    log_den = torch.logsumexp(logits.masked_fill(self_mask, float('-inf')), dim=1)
    log_prob = logits - log_den[:, None]
    per_anchor = -(log_prob.masked_fill(~positive, 0).sum(dim=1) / counts.clamp_min(1))
    
    # pseudo-label version (Eq. 17 / ldr_code_kmeans.ipynb): average over anchors that have a positive
    loss = per_anchor[valid].mean() if valid.any() else z.sum() * 0.0
    return loss, float(valid.float().mean().item())



# DP-GMM (collapsed Gibbs, NIW base) on OUTCOME features.
# Hard pseudo-labels: predict() / hard_labels_fit().  Soft targets: coclustering() + soft_target_infonce().

class CollapsedDPGMM:
    """
    DP-GMM with a Normal-Inverse-Wishart base measure, fitted by collapsed Gibbs
    sampling under the CRP prior (Experimental Approach 3).

    fit(Z)                -> runs Gibbs on teacher embeddings Z (N, d)
    coclustering(Z_new)   -> S_ij = P(c_i = c_j | E) for any batch of embeddings
    coclustering_fit()    -> S_ij for the fitted points themselves (exact Monte Carlo estimate)

    Embeddings are standardized with the fit-set mean/std before clustering, so the
    NIW prior can be set in "standardized units": the prior expected within-cluster
    std is `cluster_std` (relative to the global std of each latent dimension).

    scale="per_dim" : z-score every input dimension separately (original Exp 3 behaviour).
    scale="global"  : centre per dimension but divide by ONE scalar (RMS std over dimensions).
                      Used for PCA-projected outcome features: it keeps Euclidean distances
                      proportional to functional distances and does not blow up weak components.

    init="single"     : all points start in one cluster and Gibbs has to split it (original Exp 3).
    init="sequential" : points are added one at a time under the CRP conditional. Single-site Gibbs
                        rarely splits one big cluster when some input dimensions are weak, so this
                        is the init used for outcome features.
    """

    def __init__(self, alpha=1.0, kappa0=0.01, nu0=None, cluster_std=0.3,
                 n_burn=30, n_samples=20, thin=2, seed=0, scale="per_dim", init="single"):
        if scale not in ("per_dim", "global"):
            raise ValueError(f"Invalid scale '{scale}'")
        if init not in ("single", "sequential"):
            raise ValueError(f"Invalid init '{init}'")
        self.scale = scale
        self.init = init
        self.alpha = float(alpha)
        self.kappa0 = float(kappa0)
        self.nu0 = nu0
        self.cluster_std = float(cluster_std)
        self.n_burn, self.n_samples, self.thin = int(n_burn), int(n_samples), int(thin)
        self.rng = np.random.default_rng(seed)
        self.fitted = False

    # ---------- NIW posterior predictive (multivariate Student-t) ----------
    def _pred_params(self, n, s1, s2):
        """Student-t predictive (loc, precision-chol, logdet(scale), df) from sufficient stats."""
        d, k0, m0 = self.d, self.kappa0, self.m0
        kn, nun = k0 + n, self.nu0_ + n
        mn = (k0 * m0 + s1) / kn
        psin = self.psi0 + s2 + k0 * np.outer(m0, m0) - kn * np.outer(mn, mn)
        df = nun - d + 1.0
        scale = psin * (kn + 1.0) / (kn * df)
        L = np.linalg.cholesky(scale + 1e-9 * np.eye(d))
        Linv = np.linalg.inv(L)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        return mn, Linv, logdet, df

    @staticmethod
    def _t_logpdf(x, loc, Linv, logdet, df):
        """x: (M, d); loc: (K, d); Linv: (K, d, d); logdet, df: (K,)  ->  (M, K)."""
        d = x.shape[-1]
        diff = x[:, None, :] - loc[None]                       # (M, K, d)
        y = np.einsum("kij,mkj->mki", Linv, diff)              # whitened
        maha = (y ** 2).sum(-1)                                 # (M, K)
        return (gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * np.log(df * np.pi)
                - 0.5 * logdet - 0.5 * (df + d) * np.log1p(maha / df))

    # ---------- Gibbs ----------
    def fit(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        self.mu_ = Z.mean(0)
        if self.scale == "global":
            self.sd_ = np.full(Z.shape[1], np.sqrt(Z.var(0).mean()) + 1e-8)
        else:
            self.sd_ = Z.std(0) + 1e-8
        X = (Z - self.mu_) / self.sd_
        N, d = X.shape
        self.d = d
        self.nu0_ = float(self.nu0) if self.nu0 is not None else d + 2.0
        self.m0 = np.zeros(d)
        # E[Sigma] = Psi0 / (nu0 - d - 1)  ->  prior expected cluster cov = cluster_std^2 I
        self.psi0 = (self.cluster_std ** 2) * (self.nu0_ - d - 1.0) * np.eye(d)
        new_params = self._pred_params(0, np.zeros(d), np.zeros((d, d)))

        c = np.zeros(N, dtype=np.int64)
        stats, cache, next_label = {}, {}, 0

        def assign(i):
            # sample c_i from its conditional given the current clusters, then add x_i to that cluster
            nonlocal next_label
            x = X[i]
            labels = list(stats.keys())
            loc = np.stack([cache[k][0] for k in labels] + [new_params[0]])
            Linv = np.stack([cache[k][1] for k in labels] + [new_params[1]])
            logdet = np.array([cache[k][2] for k in labels] + [new_params[2]])
            df = np.array([cache[k][3] for k in labels] + [new_params[3]])
            log_w = np.log(np.array([stats[k][0] for k in labels] + [self.alpha], dtype=np.float64))
            logp = log_w + self._t_logpdf(x[None], loc, Linv, logdet, df)[0]
            p = np.exp(logp - logp.max()); p /= p.sum()
            j = self.rng.choice(len(p), p=p)

            if j == len(labels):                         # new component instantiated
                k_new = next_label; next_label += 1
                stats[k_new] = [0, np.zeros(d), np.zeros((d, d))]
            else:
                k_new = labels[j]
            st = stats[k_new]
            st[0] += 1; st[1] = st[1] + x; st[2] = st[2] + np.outer(x, x)
            cache[k_new] = self._pred_params(*st)
            c[i] = k_new

        if self.init == "sequential":
            for i in self.rng.permutation(N):            # CRP-order initialisation
                assign(i)
        else:
            # init: everything in one cluster (Gibbs splits it)
            stats[0] = [N, X.sum(0), X.T @ X]
            cache[0] = self._pred_params(*stats[0])
            next_label = 1

        self.samples_ = []       # retained assignment vectors (relabelled 0..K-1)
        self.sample_params_ = [] # retained per-cluster predictive params, for out-of-sample scoring
        total = self.n_burn + self.n_samples * self.thin

        for sweep in range(total):
            for i in self.rng.permutation(N):
                x, k_old = X[i], c[i]
                st = stats[k_old]
                st[0] -= 1; st[1] = st[1] - x; st[2] = st[2] - np.outer(x, x)
                if st[0] == 0:
                    del stats[k_old], cache[k_old]          # empty component removed
                else:
                    cache[k_old] = self._pred_params(*st)

                assign(i)

            if sweep >= self.n_burn and (sweep - self.n_burn) % self.thin == 0:
                labels = list(stats.keys())
                remap = {k: r for r, k in enumerate(labels)}
                self.samples_.append(np.array([remap[k] for k in c]))
                self.sample_params_.append((
                    np.stack([cache[k][0] for k in labels]),
                    np.stack([cache[k][1] for k in labels]),
                    np.array([cache[k][2] for k in labels]),
                    np.array([cache[k][3] for k in labels]),
                    np.log(np.array([stats[k][0] for k in labels], dtype=np.float64)),
                ))

        self.new_params_ = new_params
        self.samples_ = np.stack(self.samples_)               # (B, N)
        self.n_clusters_ = np.array([s.max() + 1 for s in self.samples_])

        # representative partition for HARD labels: the retained sample whose partition is closest
        # (least squares) to the posterior co-clustering matrix
        S = self.coclustering_fit()
        self.best_ = int(np.argmin([np.square((c[:, None] == c[None, :]) - S).sum() for c in self.samples_]))
        self.fitted = True
        return self

    # ---------- hard labels ----------
    def hard_labels_fit(self):
        """Cluster labels of the fitted points under the representative partition."""
        return self.samples_[self.best_]

    def predict(self, Z_new):
        """Hard label under the representative partition: the most probable existing cluster,
        or -1 when 'start a new cluster' is more probable than every existing cluster."""
        P = self.membership(Z_new)[self.best_]                      # (M, Kmax)
        labels = P.argmax(-1)
        labels[P.max(-1) < 1.0 - P.sum(-1)] = -1
        return labels

    # ---------- posterior co-clustering ----------
    def coclustering_fit(self):
        """S_ij = (1/B) sum_b 1[c_i^b == c_j^b] for the fitted points (label-permutation invariant)."""
        C = self.samples_
        return np.mean([(c[:, None] == c[None, :]) for c in C], axis=0)

    def _stack_samples(self):
        """Pad all retained samples' components into (B, Kmax) arrays for one vectorized pass."""
        B, d = len(self.sample_params_), self.d
        Kmax = max(len(p[4]) for p in self.sample_params_)
        loc = np.zeros((B, Kmax, d)); Linv = np.tile(np.eye(d), (B, Kmax, 1, 1))
        logdet = np.zeros((B, Kmax)); df = np.ones((B, Kmax)); log_w = np.full((B, Kmax), -np.inf)
        for b, (l, Li, ld, f, ln) in enumerate(self.sample_params_):
            K = len(ln)
            loc[b, :K], Linv[b, :K], logdet[b, :K], df[b, :K], log_w[b, :K] = l, Li, ld, f, ln
        loc, Linv, logdet, df = loc.reshape(-1, d), Linv.reshape(-1, d, d), logdet.ravel(), df.ravel()
        # whitening as one matmul: Linv_k (x - loc_k) = x @ A.T - off_k, with A the stacked Linv rows
        A = Linv.reshape(-1, d)                                   # (B*Kmax*d, d)
        off = np.einsum("kij,kj->ki", Linv, loc)                  # (B*Kmax, d)
        const = gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * np.log(df * np.pi) - 0.5 * logdet
        self._stack = (A, off, const, df, log_w, B, Kmax)
        nl, nLi, nld, ndf = [np.asarray(a) for a in self.new_params_]
        self._new = (nl, nLi, float(gammaln((ndf + d) / 2) - gammaln(ndf / 2)
                                    - 0.5 * d * np.log(ndf * np.pi) - 0.5 * nld), float(ndf))

    def membership(self, Z_new):
        """P_b(c=k | z) under each retained sample b, for existing clusters only (new/singleton dropped
        after normalization). Returns (B, M, Kmax); padded components have probability 0."""
        if not hasattr(self, "_stack"):
            self._stack_samples()
        A, off, const, df, log_w, B, Kmax = self._stack
        X = (np.asarray(Z_new, dtype=np.float64) - self.mu_) / self.sd_
        M, d = X.shape
        y = (X @ A.T).reshape(M, B * Kmax, d) - off[None]                                    # whitened
        ll = const - 0.5 * (df + d) * np.log1p((y ** 2).sum(-1) / df)                       # (M, B*Kmax)
        ll = ll.reshape(M, B, Kmax).transpose(1, 0, 2)                                       # (B, M, Kmax)
        logp = log_w[:, None, :] + ll
        nl, nLi, nconst, ndf = self._new
        maha_new = (((X - nl) @ nLi.T) ** 2).sum(-1)
        ll_new = np.log(self.alpha) + nconst - 0.5 * (ndf + d) * np.log1p(maha_new / ndf)     # (M,)
        mx = np.maximum(logp.max(-1), ll_new[None])                                              # (B, M)
        p = np.exp(logp - mx[..., None])
        z = p.sum(-1) + np.exp(ll_new[None] - mx)
        return p / z[..., None]

    def coclustering(self, Z_new):
        """S_ij = (1/B) sum_b sum_k P_b(c_i=k) P_b(c_j=k). A 'new' assignment is a singleton,
        so it contributes nothing to co-clustering."""
        P = self.membership(Z_new)                                  # (B, M, Kmax)
        Q = P.transpose(1, 0, 2).reshape(P.shape[1], -1)            # (M, B*Kmax)
        return (Q @ Q.T) / P.shape[0]


def soft_target_infonce(z, S, temperature, min_positive_mass=0.5):
    """L_DP-Con: InfoNCE with soft positives S_bar_ij = S_ij / sum_{l!=i} S_il (S detached)."""
    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)

    S = S.masked_fill(self_mask, 0.0)
    mass = S.sum(dim=1)
    valid = mass > min_positive_mass                     # anchors in I
    target = S / mass.clamp_min(1e-8)[:, None]

    log_den = torch.logsumexp(logits.masked_fill(self_mask, float("-inf")), dim=1)
    log_prob = (logits - log_den[:, None]).masked_fill(self_mask, 0.0)
    per_anchor = -(target * log_prob).sum(dim=1)

    if not valid.any():
        return z.sum() * 0.0, 0.0
    return per_anchor[valid].mean(), float(valid.float().mean().item())


class LDRAgent:

    def __init__(self, env, env_ID, config=None, device=None):

        self.env = env
        self.env_ID = env_ID
        self.env_name = self.env_ID.split("-")[0].lower()  
        self.config = config or LDRConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu" ))
        self.num_envs = env.num_envs
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32)
        
        # # Safety checks for action space
        # assert isinstance(env.action_space, gym.spaces.Box)
        # assert np.all(np.isfinite(self.action_low))
        # assert np.all(np.isfinite(self.action_high))
        # assert np.all(self.action_high > self.action_low)

        # Training counters
        self.num_timesteps = 0
        self.num_updates = 0

        # Track running returns for each parallel environment.
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.completed_returns = []

        self.last_obs = None #current observations
        self.target_entropy = -float(self.action_dim) # SAC entropy target





        # Actor and Critic Networks 
        input_dim = self.obs_dim + self.config.latent_dim

        self.actor = Actor(
            obs_dim=input_dim, 
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim, 
            latent_dim=self.config.latent_dim
            ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate) 

        self.critic = Critic(
            obs_dim=input_dim, 
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim, 
            latent_dim=self.config.latent_dim
            ).to(self.device)

        self.critic_target = Critic(
            obs_dim=input_dim, 
            action_dim=self.action_dim, 
            hidden_dim=self.config.hidden_dim, 
            latent_dim=self.config.latent_dim
            ).to(self.device)

        # Initially copying online weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target network is not trained by gradient descent rather updated from online Critic using Polyak averaging
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)






        # Encoder & Decoder
        self.encoder = DynamicsEncoder(
            self.obs_dim, 
            self.action_dim, 
            self.config.n_past, 
            self.config.latent_dim,
            reg_mode = self.config.encoder_reg_mode,
            ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.learning_rate)

        self.decoder = DynamicsDecoder(
            self.obs_dim, 
            self.action_dim, 
            self.config.latent_dim, 
            self.config.decoder_hidden_dim
            ).to(self.device)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.config.learning_rate)

        # EMA teachers (phi_bar, omega_bar): never trained by gradients, used only to build outcome features
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_decoder = copy.deepcopy(self.decoder)
        for net in (self.teacher_encoder, self.teacher_decoder):
            net.requires_grad_(False)
            net.eval()
        self.probe_obs = None        # (M, obs_dim)    shared reference states
        self.probe_actions = None    # (M, action_dim) shared reference actions
        self.pca_mean = None         # (M * obs_dim,)
        self.pca_comp = None         # (d_outcome, M * obs_dim)

        # DP-GMM posterior (refit periodically) and bookkeeping
        self.dpgmm = None
        self.next_dp_refresh = None
        self.dp_log = []
        self.timing = {"explore": 0.0, "dp_refresh": 0.0, "optimize": 0.0, "dp_targets": 0.0, "evaluate": 0.0}


        self.history_queue = HistoryQueue(self.num_envs, self.config.n_past, self.obs_dim, self.action_dim)
        
        self.replay_buffer = ReplayBuffer(
            buffer_size = self.config.buffer_size, 
            num_envs = self.num_envs, 
            obs_dim = self.obs_dim, 
            action_dim = self.action_dim, 
            device = self.device, 
            n_past = self.config.n_past
            )

        # Learnable log entropy coefficient
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)


    # polyak averaging for target network update
    def update_target(self):
        tau = self.config.tau
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.mul_(1 - tau)
                target_param.add_(tau * param)

    # phi_bar <- rho * phi_bar + (1 - rho) * phi ;  omega_bar <- rho * omega_bar + (1 - rho) * omega
    @torch.no_grad()
    def update_teacher(self):
        rho = self.config.ema_rho
        for teacher, student in ((self.teacher_encoder, self.encoder),
                                 (self.teacher_decoder, self.decoder)):
            for t, p in zip(teacher.parameters(), student.parameters()):
                t.mul_(rho).add_((1.0 - rho) * p)
            for t, b in zip(teacher.buffers(), student.buffers()):
                t.copy_(b)        # spectral-norm u/v buffers and the decoder's target_scaler stats

    @torch.no_grad()
    def raw_outcomes(self, history):
        """y_i = concat_m Delta s_hat(s_m, a_m, z_tilde_i): teacher-decoder mean (normalized state
        change) on the shared probes, conditioned on the teacher latent.  Shape (n, M * obs_dim)."""
        history = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        M, out = self.probe_obs.shape[0], []
        for h in history.split(self.config.outcome_chunk):
            zbar = self.teacher_encoder(h).loc                                 # (b, d_z)
            b = zbar.shape[0]
            s = self.probe_obs.unsqueeze(0).expand(b, M, -1).reshape(b * M, -1)
            a = self.probe_actions.unsqueeze(0).expand(b, M, -1).reshape(b * M, -1)
            z = zbar.unsqueeze(1).expand(b, M, -1).reshape(b * M, -1)
            out.append(self.teacher_decoder(s, a, z).loc.reshape(b, M * self.obs_dim))
        return torch.cat(out)

    @torch.no_grad()
    def outcome_features(self, history):
        """PCA-projected outcome vectors: the points the DP-GMM is fitted on / scored against."""
        y = self.raw_outcomes(history)
        return ((y - self.pca_mean) @ self.pca_comp.T).cpu().numpy()

    def refresh_clusters(self):
        """Resample probes, refit the outcome PCA, then DP-GMM inference in OUTCOME space."""
        cfg = self.config

        # shared reference set, fixed until the next refresh
        probes = self.replay_buffer.sample(cfg.num_probes)
        self.probe_obs, self.probe_actions = probes["obs"], probes["actions"]

        # PCA of the (M * obs_dim) outcome vector: keep the fewest dims reaching outcome_pca_var.
        # The outcome dimension differs per environment (obs_dim 11 / 17 / 27), the kept dims adapt to it.
        histories, ids = self.replay_buffer.sample_histories(cfg.dp_num_points)
        Y = self.raw_outcomes(histories).cpu().numpy()
        pca = PCA(n_components=min(cfg.outcome_pca_dim, Y.shape[0] - 1, Y.shape[1])).fit(Y)
        cum = np.nan_to_num(np.cumsum(pca.explained_variance_ratio_), nan=1.0)
        n_dim = int(min(np.searchsorted(cum, cfg.outcome_pca_var) + 1, len(cum)))
        self.pca_mean = torch.as_tensor(pca.mean_, dtype=torch.float32, device=self.device)
        self.pca_comp = torch.as_tensor(pca.components_[:n_dim], dtype=torch.float32, device=self.device)

        y = self.outcome_features(histories)
        self.dpgmm = CollapsedDPGMM(
            alpha=cfg.dp_alpha, kappa0=cfg.dp_kappa0, nu0=cfg.dp_nu0,
            cluster_std=cfg.dp_cluster_std, n_burn=cfg.dp_burn,
            n_samples=cfg.dp_samples, thin=cfg.dp_thin,
            seed=cfg.seed + self.num_timesteps,
            scale="global", init="sequential",
        ).fit(y)

        # diagnostic only: agreement with the (privileged) dynamics IDs -- never used in training
        fit_labels = self.dpgmm.hard_labels_fit() if cfg.dp_hard_labels else self.dpgmm.samples_[-1]
        ari = adjusted_rand_score(ids, fit_labels)
        entry = {"timesteps": self.num_timesteps,
                 "K_used": int(fit_labels.max() + 1),
                 "K_mean": float(self.dpgmm.n_clusters_.mean()),
                 "K_min": int(self.dpgmm.n_clusters_.min()),
                 "K_max": int(self.dpgmm.n_clusters_.max()),
                 "n_true_ids": int(len(np.unique(ids))),
                 "ari_vs_ids": float(ari),
                 "outcome_dims": n_dim,
                 "outcome_pca_var": float(cum[n_dim - 1])}
        self.dp_log.append(entry)
        print(f"[DP-GMM] step {self.num_timesteps}: K={entry['K_used']} "
              f"(posterior {entry['K_min']}-{entry['K_max']}), true IDs in subset {entry['n_true_ids']}, "
              f"ARI {ari:.3f}, outcome dims {n_dim} (var {entry['outcome_pca_var']:.3f})")

    # Convert normalized actions [-1, 1] to environment bounds
    def _toEnvAction(self, action):
        return (self.action_low + 0.5 * (self.action_high - self.action_low) * (action + 1.0))

    # Convert environment actions back to [-1, 1].
    def _reverseEnvAction(self, action):
        return (2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0)

    
    
    def predict(self, obs, history=None, deterministic=True, train_dropout=False):

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device )

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if history is None:
            history = self.history_queue.get()

        self.actor.eval()
        self.encoder.eval()

        with torch.no_grad():
            z = self._latent(history, train_dropout=train_dropout)
            x = torch.cat([obs, z], dim=-1)
            action = self.actor.predict(x, deterministic=deterministic)

        return self._toEnvAction(action.cpu().numpy())


    def _latent(self, history, train_dropout=False):

        history = torch.as_tensor(history, dtype=torch.float32, device=self.device )

        if history.ndim == 2:
            history = history.unsqueeze(0)

        with torch.no_grad():
            z = self.encoder(history).loc

            if train_dropout and self.config.latent_dropout_prob > 0:
                keep = (
                    torch.rand((z.shape[0], 1), device=self.device)
                    >= self.config.latent_dropout_prob
                ).float()
                z = z * keep

        return z


    def _selectAction(self, obs, deterministic=False):

        if self.num_timesteps < self.config.learning_starts:

            # Random normalized actions during warm-up
            action = np.random.uniform(-1.0, 1.0, size=(self.num_envs, self.action_dim) ).astype(np.float32)

        else:
            # Predict environment-scaled actions
            env_action = self.predict(obs, deterministic=deterministic, train_dropout=not deterministic)

            # Convert back to normalized [-1, 1]
            action = self._reverseEnvAction(env_action)

        return np.clip(action, -1.0, 1.0).astype(np.float32)



    # Collect transitions using vectorized environments
    def explore(self):

        if self.last_obs is None:
            self.last_obs = self.env.reset()

        collected_steps = 0

        while collected_steps < self.config.train_freq:

            history=self.history_queue.get()  # history BEFORE current action
            ids= self.env.dynamics_ids.copy()  # capture pre-step dynamics IDs (auto-reset may change them)

            buffer_actions = self._selectAction(self.last_obs, deterministic=False)
            env_actions = self._toEnvAction(buffer_actions) #scaling to environment action space before applying

            new_obs, rewards, dones, infos = self.env.step(env_actions)

            self.episode_returns += np.asarray(rewards, dtype=np.float64)
            
            for i in np.flatnonzero(dones):
                self.completed_returns.append(float(self.episode_returns[i]))
                self.episode_returns[i] = 0.0

            # 4. Identify time-limit truncations
            timeouts = np.array(
                [
                    info.get("TimeLimit.truncated", False)
                    for info in infos
                ],
                dtype=np.float32
            )

            # 5. Preserve actual terminal observations
            buffer_next_obs = new_obs.copy()

            for i in range(self.num_envs):
                if dones[i]:
                    buffer_next_obs[i] = infos[i]["terminal_observation"]

            
            self.replay_buffer.push(
                obs=self.last_obs,
                next_obs=buffer_next_obs,
                actions=buffer_actions,
                rewards=rewards,
                dones=dones,
                timeouts=timeouts,
                history = history,
                dynamics_ids=ids
            )

            self.history_queue.append(self.last_obs, buffer_actions)
            self.history_queue.reset_done(dones, new_obs)
            self.last_obs = new_obs.copy()
            self.num_timesteps += self.num_envs
            collected_steps += self.num_envs

        return collected_steps

    
    def optimize(self):

        self.actor.train()
        self.critic.train()
        self.encoder.train()
        self.decoder.train()

        metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "alpha": [],
            "vae_loss": [],
            "contrastive_loss": [],
            "positive_fraction": [],
            "dp_clusters": [],
            "dp_unassigned": []
        }

        for _ in range(self.config.gradient_steps):

            batch = self.replay_buffer.sample(self.config.batch_size)

            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_obs = batch["next_obs"]
            dones = batch["dones"]

            # LDR-specific
            history = batch["history"]
            dynamics_ids = batch["ids"]

            
            qz = self.encoder(history)
            z = qz.rsample()

            # Latent dropout
            keep = (torch.rand((z.shape[0], 1), device=self.device)  >=   self.config.latent_dropout_prob).float()
            z_rl = z * keep # sampled latent including dropout for RL training

            # Condition observations on latent dynamics
            obs_z = torch.cat([obs, z_rl], dim=-1)
            next_obs_z = torch.cat([next_obs, z_rl.detach()], dim=-1)

            # entropy coefficient
            alpha = self.log_alpha.exp().detach()

            # training critic with target Q-values
            with torch.no_grad():

                next_actions, next_log_prob = (self.actor.actionLogProb(next_obs_z) )
                next_q1, next_q2 = self.critic_target(next_obs_z, next_actions)
                next_q = (torch.min(next_q1, next_q2) - alpha * next_log_prob)

                target_q = (rewards * self.config.reward_scale + self.config.gamma * (1.0 - dones) * next_q )

            
            q1, q2 = self.critic(obs_z.detach(), actions)
            critic_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # training actor with policy gradient
            new_actions, log_prob = (self.actor.actionLogProb(obs_z)  )

            # Freeze critic temporarily
            # for param in self.critic.parameters():
            #     param.requires_grad_(False)

            # (no critic freeze, as in Xu: gradients the actor loss leaves on the critic
            #  are cleared by critic_optimizer.zero_grad() before the next critic update)
            q1_pi, q2_pi = self.critic(obs_z, new_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (alpha * log_prob - min_q_pi).mean()



            # VAE reconstruction loss
            recon_loss = -self.decoder.logProb(obs, next_obs, actions, z).mean()

            # KL divergence 
            kl_loss = 0.5 * (qz.loc.square() + qz.scale.square() - 1.0 - 2.0 * qz.scale.log()).sum(dim=-1).mean()
            
            progress = min(self.num_timesteps / self.config.total_timesteps, 1.0)
            beta = self.config.beta_max * min(progress / self.config.beta_anneal_fraction, 1.0)
            vae_loss = recon_loss + beta * kl_loss


            # Contrastive loss (InfoNCE) for latent dynamics
            contrastive_loss = z.sum() * 0.0
            positive_fraction = 0.0

            # Cluster-based contrastive loss in OUTCOME space. dynamics_ids are NOT used.
            dp_unassigned = 0.0
            if self.dpgmm is not None:
                _t = time.perf_counter()
                y_tilde = self.outcome_features(history)
                if self.config.dp_hard_labels:
                    # hard pseudo-labels: same cluster -> positive pair (as with K-means)
                    labels = self.dpgmm.predict(y_tilde)
                    new = labels < 0
                    labels[new] = -1 - np.flatnonzero(new)       # fits no cluster: unique label -> no positives
                    dp_unassigned = float(new.mean())
                    pseudo_labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
                    contrastive_loss, positive_fraction = multi_positive_infonce(
                        z, pseudo_labels, self.config.infonce_temp
                    )
                else:
                    # soft targets: posterior co-clustering probabilities as positive-pair weights
                    S = torch.as_tensor(self.dpgmm.coclustering(y_tilde), dtype=torch.float32, device=self.device)
                    contrastive_loss, positive_fraction = soft_target_infonce(
                        z, S, self.config.infonce_temp, self.config.dp_min_positive_mass
                    )
                self.timing["dp_targets"] += time.perf_counter() - _t

            # Combined total loss for actor, encoder, and decoder
            total_loss = (actor_loss + self.config.vae_loss_weight * vae_loss + self.config.contrastive_loss_weight * contrastive_loss)

            self.actor_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            total_loss.backward()

            self.actor_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # teacher (encoder + decoder) EMA update after the student update
            if self.dpgmm is not None:
                self.update_teacher()

            # Restore critic gradients
            # for param in self.critic.parameters():
            #     param.requires_grad_(True)

            # Update entropy coefficient
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # update target networks using Polyak averaging
            self.num_updates += 1

            if (self.num_updates % self.config.target_update_interval == 0):
                self.update_target()

            # Saving metrics for logging
            metrics["actor_loss"].append(actor_loss.item())
            metrics["critic_loss"].append(critic_loss.item())
            metrics["alpha_loss"].append(alpha_loss.item())
            metrics["alpha"].append(alpha.item())
            metrics["vae_loss"].append(vae_loss.item())
            metrics["contrastive_loss"].append(contrastive_loss.item())

            if torch.is_tensor(positive_fraction):
                positive_fraction = positive_fraction.item()

            metrics["positive_fraction"].append(float(positive_fraction) )
            metrics["dp_clusters"].append(float(self.dp_log[-1]["K_used"]) if self.dpgmm is not None else 0.0)
            metrics["dp_unassigned"].append(dp_unassigned)

        return {key: float(np.mean(values)) for key, values in metrics.items()}


    # Evaluates Deterministic LDR policy
    def evaluate(self, eval_env, eval_num=1, seed=None):


        self.actor.eval()
        self.encoder.eval()

        num_envs = eval_env.num_envs
        all_returns = []

        for episode in range(eval_num):

            if seed is not None:
                eval_env.seed(seed + episode * num_envs)

            obs = eval_env.reset()

            
            history = HistoryQueue(
                num_envs,
                self.config.n_past,
                self.obs_dim,
                self.action_dim
            )
            history.reset_done(np.ones(num_envs, dtype=bool), obs)

            episode_returns = np.zeros(num_envs, dtype=np.float64)

            # Track which environments have finished
            finished = np.zeros(num_envs, dtype=bool)

            while not np.all(finished):

                # LDR: predict using current obs and past history
                actions = self.predict(
                    obs,
                    history=history.get(),
                    deterministic=True
                )

                next_obs, rewards, dones, infos = eval_env.step(actions)

                episode_returns[~finished] += rewards[~finished]

                # LDR specific: store normalized actions in history
                normalized_actions = self._reverseEnvAction(actions)

                history.append(obs, normalized_actions)
                history.reset_done(dones, next_obs) 

                obs = next_obs
                finished |= dones

            all_returns.extend(episode_returns.tolist())

        return float(np.mean(all_returns))



    def train(self, eval_env=None, eval_num=1, RUN_DIR=None):

        RUN_DIR = Path(RUN_DIR or f"{self.env_name}_ldr_dpgmm_results")
        RUN_DIR.mkdir(parents=True, exist_ok=True)

        self.env.seed(self.config.seed)
        self.last_obs = self.env.reset()

        # LDR-specific: Reset history queue
        # self.history_queue.reset_done(np.ones(self.num_envs, dtype=bool))
        self.history_queue.reset_done(np.ones(self.num_envs, dtype=bool), self.last_obs)

        # Reset counters and reward tracking
        self.num_timesteps = 0
        self.num_updates = 0

        self.episode_returns.fill(0.0)
        self.completed_returns.clear()
        self.dpgmm, self.next_dp_refresh = None, None
        self.probe_obs = self.probe_actions = self.pca_mean = self.pca_comp = None
        self.dp_log.clear()
        self.timing = {k: 0.0 for k in self.timing}

        # Evaluation tracking
        best_eval_reward = float("-inf")

        next_eval_step = self.config.eval_interval

        training_history = []

        while self.num_timesteps < self.config.total_timesteps:

            _t = time.perf_counter()
            collected_steps = self.explore()
            self.timing["explore"] += time.perf_counter() - _t

            metrics = None

            # end of warm-up -> init teachers from students, first DP-GMM fit; then periodic refresh
            _t = time.perf_counter()
            if self.num_timesteps >= self.config.contrastive_start_iterations:
                if self.dpgmm is None:
                    self.teacher_encoder.load_state_dict(self.encoder.state_dict())
                    self.teacher_decoder.load_state_dict(self.decoder.state_dict())
                    self.refresh_clusters()
                    self.next_dp_refresh = self.num_timesteps + self.config.dp_refresh_interval
                elif self.num_timesteps >= self.next_dp_refresh:
                    self.refresh_clusters()
                    self.next_dp_refresh += self.config.dp_refresh_interval
            self.timing["dp_refresh"] += time.perf_counter() - _t

            if (self.num_timesteps >= self.config.learning_starts) and (len(self.replay_buffer) >= self.config.batch_size):
                _t = time.perf_counter()
                metrics = self.optimize()
                self.timing["optimize"] += time.perf_counter() - _t

            if metrics is not None:
                metrics["timesteps"] = self.num_timesteps
                training_history.append(metrics)

            if (eval_env is not None) and (self.num_timesteps >= next_eval_step):

                _t = time.perf_counter()
                eval_reward = self.evaluate(eval_env, eval_num=eval_num, seed=0)
                self.timing["evaluate"] += time.perf_counter() - _t
                print(f"Steps: {self.num_timesteps} | Eval reward: {eval_reward:.2f}")

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save(RUN_DIR / f"{self.env_name}_dpgmm_best.pt")
                    print(f"New best reward: {best_eval_reward:.2f}")

                # Advance evaluation threshold
                while next_eval_step <= self.num_timesteps:
                    next_eval_step += self.config.eval_interval

        print("Training completed! seconds per phase:", {k: round(v, 1) for k, v in self.timing.items()})

        return training_history


    
    
    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),

            # LDR networks
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "teacher_encoder": self.teacher_encoder.state_dict(),
            "teacher_decoder": self.teacher_decoder.state_dict(),

            # outcome-space state (the fitted DP-GMM itself is refit at the next refresh)
            "probe_obs": self.probe_obs,
            "probe_actions": self.probe_actions,
            "pca_mean": self.pca_mean,
            "pca_comp": self.pca_comp,

            # SAC optimizers
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),

            # LDR optimizers
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),

            # Entropy coefficient
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),

            # Training progress
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,

            # Configuration
            "config": asdict(self.config)

        }, path)


    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        # SAC networks
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(
            checkpoint["critic_target"]
        )

        # LDR networks
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.teacher_encoder.load_state_dict(checkpoint.get("teacher_encoder", self.encoder.state_dict()))
        self.teacher_decoder.load_state_dict(checkpoint.get("teacher_decoder", self.decoder.state_dict()))

        def to_dev(x):
            return None if x is None else x.to(self.device)
        self.probe_obs = to_dev(checkpoint.get("probe_obs"))
        self.probe_actions = to_dev(checkpoint.get("probe_actions"))
        self.pca_mean = to_dev(checkpoint.get("pca_mean"))
        self.pca_comp = to_dev(checkpoint.get("pca_comp"))

        # SAC optimizers
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )
        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        # LDR optimizers
        self.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer"]
        )
        self.decoder_optimizer.load_state_dict(
            checkpoint["decoder_optimizer"]
        )

        # Entropy coefficient
        with torch.no_grad():
            self.log_alpha.copy_(
                checkpoint["log_alpha"].to(self.device)
            )

        self.alpha_optimizer.load_state_dict(
            checkpoint["alpha_optimizer"]
        )

        # Training progress
        self.num_timesteps = checkpoint["num_timesteps"]
        self.num_updates = checkpoint["num_updates"]

        # Target critic is not trained directly
        self.critic_target.eval()




class makeVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.dynamics_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def seed(self, seed):
        self._seed = seed

    def reset(self):
        observations = []
        for i, env in enumerate(self.envs):
            seed = getattr(self, "_seed", None)
            obs, _ = env.reset(
                seed=None if seed is None else seed + i
            )
            observations.append(obs)
        self._seed = None
        return np.stack(observations).astype(np.float32)

    def step(self, actions):
        observations, rewards, dones, infos = [], [], [], []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            info = dict(info)
            info["TimeLimit.truncated"] = truncated and not terminated

            if done:
                info["terminal_observation"] = obs.copy()
                obs, _ = env.reset()

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos
        )

    def close(self):
        for env in self.envs:
            env.close()


PHYSICS_SPECS = {   # Xu: local_wrappers/physics_wrapper.py
    "Hopper":      dict(friction_range=None,       root_dof=3),
    "Walker2d":    dict(friction_range=None,       root_dof=3),
    "HalfCheetah": dict(friction_range=(0.4, 1.0), root_dof=3),
    "Ant":         dict(friction_range=(0.2, 1.0), root_dof=6),
}


class PhysicsWrapper(gym.Wrapper):
    def __init__(self, env, mode="custom", mass_scale=1.0, damping_scale=1.0, torque_scale=1.0,
                 friction_scale=1.0, multi_episode_dynamics=100, required_experience_length=None,
                 mass_range=(0.5, 2.0), damping_range=(0.5, 2.0), torque_range=(0.5, 1.5),
                #  mass_range=(0.8, 1.5), damping_range=(0.8, 1.5), torque_range=(0.8, 1.2),
                 friction_range=None, root_dof=3):
        super().__init__(env)
        self.mode = mode
        self.model = self.unwrapped.model
        self.base_mass = self.model.body_mass.copy()
        self.base_damping = self.model.dof_damping.copy()
        self.base_friction = self.model.geom_friction.copy()
        self.friction_range = None if friction_range is None else tuple(friction_range)
        self.root_dof = root_dof
        self.mass_range, self.damping_range, self.torque_range = map(tuple, (mass_range, damping_range, torque_range))
        self.multi_episode_dynamics = int(multi_episode_dynamics)
        self.required_experience_length = required_experience_length
        self.steps_since_change, self.episodes = 0, 0
        self.last_dynamics_changed = False
        self.torque_scale = 1.0
        self._rng = np.random.default_rng()
        if mode == "custom":
            fric = self.base_friction.copy(); fric[:, 0] *= friction_scale
            self._set_physics(self.base_mass * mass_scale, self.base_damping * damping_scale, torque_scale, fric)

    def _set_physics(self, mass, damping, torque_scale, friction=None):
        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping
        if friction is not None:
            self.model.geom_friction[:] = friction
        self.torque_scale = float(torque_scale)

    def _sample_physics(self):
        # same draw order as Xu: friction, mass, damping, torque
        fric = None
        if self.friction_range is not None:
            fric = self.base_friction.copy()
            fric[:, 0] *= self._rng.uniform(*self.friction_range, size=fric.shape[0])
        self._set_physics(
            self.base_mass * self._rng.uniform(*self.mass_range, size=self.base_mass.shape),
            self.base_damping * self._rng.uniform(*self.damping_range, size=self.base_damping.shape),
            self._rng.uniform(*self.torque_range), fric)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.mode == "change" and self.required_experience_length is not None:
            # Xu: start nominal, switch once enough steps have been collected
            changed = self.steps_since_change >= self.required_experience_length
        else:
            changed = (
                (self.mode == "anchor" and self.episodes == 0)
                or (self.mode == "change" and self.episodes % self.multi_episode_dynamics == 0)
            )

        if changed:
            self._sample_physics()
            self.steps_since_change = 0

        obs, info = self.env.reset(**kwargs)
        self.episodes += 1
        self.last_dynamics_changed = changed
        return obs, {**info, "dynamics_changed": changed}

    def step(self, action):
        self.steps_since_change += 1
        return self.env.step(np.asarray(action) * self.torque_scale)

    def _param_vec(self, fric, mass, damp, torque):
        parts = [fric[:, 0].copy()] if self.friction_range is not None else []
        return np.concatenate(parts + [mass[1:].copy(), damp[self.root_dof:].copy(), np.array([torque])])

    def getRandomizedParameters(self):
        return self._param_vec(self.model.geom_friction, self.model.body_mass, self.model.dof_damping, self.torque_scale)

    def getNominalParameters(self):
        return self._param_vec(self.base_friction, self.base_mass, self.base_damping, 1.0)

    


class CleanObservationWrapper(gym.ObservationWrapper):
    """Ant: keep qpos[2:]+qvel (27 dims), drop the 78 cfrc_ext dims Ant-v5 includes by default."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float64)
    def observation(self, obs):
        return obs[:27]

class BodyFrameWrapper(gym.ObservationWrapper):
    """Ant: rotate root linear velocity (obs[13:16]) into the body frame."""
    def observation(self, obs):
        w, x, y, z = obs[1:5]
        new = obs.copy()
        new[13:16] = R.from_quat([x, y, z, w]).inv().apply(obs[13:16])
        return new

class LimitAngleTorqueWrapper(gym.Wrapper):
    """Walker2d: clip ankle ('foot') actuators to 20% of ctrlrange."""
    def __init__(self, env):
        super().__init__(env)
        m = self.unwrapped.model
        for i in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) or ""
            if "foot" in name:
                lim = 0.2 * m.actuator_ctrlrange[i, 1]
                m.actuator_ctrlrange[i] = [-lim, lim]



class GlobalDynamicsIDWrapper:
    """Persistent dynamics IDs: identical parameter vectors share an ID across all modes."""

    MANAGER_DYNAMICS_ID_STR = "dynamics_ids"

    def __init__(self, venv):
        self.venv = venv
        self.envs = venv.envs
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.physics_envs = [self._find_physics_wrapper(env) for env in self.envs]

        # the initial dynamics ID for each environment is set to 0
        self.dynamics_ids = np.zeros(self.num_envs, dtype=np.int64)

        nominal_params = tuple(self.physics_envs[0].getNominalParameters().tolist())
        for physics in self.physics_envs[1:]:
            if tuple(physics.getNominalParameters().tolist()) != nominal_params:
                raise ValueError("All environments must share the same nominal physics")

        self.records = pd.DataFrame({
            "dynamics_id": pd.Series([0], dtype="int64"),
            "parameters": [nominal_params],
        })
        self._initialized = False

    @staticmethod
    def _find_physics_wrapper(env):
        while isinstance(env, gym.Wrapper):
            if isinstance(env, PhysicsWrapper):
                return env
            env = env.env
        raise ValueError("Every environment must contain PhysicsWrapper")

    def seed(self, seed):
        return self.venv.seed(seed)

    def _get_dynamics_id(self, physics):
        params = tuple(physics.getRandomizedParameters().tolist())
        matches = self.records.loc[
            self.records["parameters"].map(lambda p: p == params),
            "dynamics_id",
        ]
        if not matches.empty:
            return int(matches.iloc[0])

        dynamics_id = int(self.records["dynamics_id"].max()) + 1
        self.records.loc[len(self.records)] = [dynamics_id, params]
        return dynamics_id

    def reset(self):
        obs = self.venv.reset()
        for i, physics in enumerate(self.physics_envs):
            if not self._initialized or physics.last_dynamics_changed:
                self.dynamics_ids[i] = self._get_dynamics_id(physics)
        self._initialized = True
        return obs

    def step(self, actions):
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        old_ids = self.dynamics_ids.copy()
        obs, rewards, dones, infos = self.venv.step(actions)
        # makeVecEnv automatically resets done environments.
        for i in np.flatnonzero(dones):
            if self.physics_envs[i].last_dynamics_changed:
                self.dynamics_ids[i] = self._get_dynamics_id(self.physics_envs[i])
        for i, info in enumerate(infos):
            info[self.MANAGER_DYNAMICS_ID_STR] = int(old_ids[i])
            info["next_dynamics_id"] = int(self.dynamics_ids[i])
        return obs, rewards, dones, infos

    def save_dynamics_log(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.records.copy()
        df["parameters"] = df["parameters"].apply(json.dumps)
        df.to_csv(path, index=False)

    def close(self):
        return self.venv.close()

    def __getattr__(self, name):
        return getattr(self.venv, name)



ENV_ID = "HalfCheetah-v5"          # or "Walker2d-v5", "Ant-v5", "Hopper-v5" "HalfCheetah-v5"
TRAIN_SWITCH = {"HalfCheetah-v5": 5000}.get(ENV_ID, 2000)

NUM_ENVS = 16

def make_env(env_id, required_experience_length, mode="change", render_mode=None, **physics_kwargs):
    family = env_id.split("-")[0]
    def _make():
        env = gym.make(env_id, render_mode=render_mode)
        if family == "Ant":
            env = CleanObservationWrapper(env)
        env = PhysicsWrapper(env, mode=mode, required_experience_length=required_experience_length,
                             **PHYSICS_SPECS[family], **physics_kwargs)
        if family == "Ant":
            env = BodyFrameWrapper(env)
        if family == "Walker2d":
            env = LimitAngleTorqueWrapper(env)
        return env
    return _make

train_env = GlobalDynamicsIDWrapper(makeVecEnv([make_env(ENV_ID, TRAIN_SWITCH) for _ in range(NUM_ENVS)]))
eval_env  = GlobalDynamicsIDWrapper(makeVecEnv([make_env(ENV_ID, 1) for _ in range(NUM_ENVS)]))

config = LDRConfig(**ENV_PRESETS[ENV_ID])

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

agent = LDRAgent(env=train_env, env_ID=ENV_ID, config=config)



# Main training

RUN_DIR = Path(f"{agent.env_name}_dpgmm_results")
RUN_DIR.mkdir(exist_ok=True, parents=True)

try:
    # xu uses eval_num = 1, ours is averaging over 3
    history = agent.train(eval_env=eval_env, eval_num=3, RUN_DIR=RUN_DIR) 
    agent.save(str(RUN_DIR / f"{agent.env_name}_dpgmm_final.pt"))
    history_df = pd.DataFrame(history)
    history_df.to_csv(RUN_DIR / "training_history.csv", index=False)
    print("Final checkpoint:", RUN_DIR / f"{agent.env_name}_dpgmm_final.pt")
finally:
    train_env.close()
    eval_env.close()

print("Finished training episodes:", len(agent.completed_returns))
print("Optimization records:", len(history))

# DP-GMM refresh log (cluster count and ARI vs. true dynamics IDs, diagnostic only)
pd.DataFrame(agent.dp_log).to_csv(RUN_DIR / "dp_log.csv", index=False)
print(pd.DataFrame(agent.dp_log))