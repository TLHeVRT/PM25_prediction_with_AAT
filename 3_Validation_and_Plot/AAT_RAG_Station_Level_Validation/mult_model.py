import numpy as np
import torch
import torch.nn as nn

HOURS_PER_YEAR = 365 * 24


class StaticProfileEncoder(nn.Module):
    """Static profile encoder."""
    def __init__(self, in_features, d_profile=128, dropout=0.2):
        super(StaticProfileEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_profile)
        )

    def forward(self, static_features):
        return self.net(static_features)

class CrossSequenceBlock(nn.Module):
    """Column-wise Cross-Attention block."""

    def __init__(self, d_model=64, n_heads=8, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, q, kv):
        # q: [B*T, 1, d_model]
        # kv: [B*T, 10, d_model]
        q_norm = self.norm1(q)
        kv_norm = self.norm1(kv)

        attn_out, _ = self.cross_attn(
            q_norm, kv_norm, kv_norm, need_weights=False
        )
        q = q + attn_out

        q = q + self.ffn(self.norm2(q))
        return q


class Full2DBlock(nn.Module):
    """Full 2D Alternating Attention block."""

    def __init__(self, d_model=64, n_heads=8, dropout=0.2):
        super().__init__()
        # Time-wise Transformer
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        # Column-wise Transformer
        self.col_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )

    def forward(self, grid):
        # grid shape: [B, 11, 192, d_model]
        B, N, T, D = grid.shape

        grid_time = grid.reshape(B * N, T, D)
        grid_time = self.time_attn(grid_time)
        grid = grid_time.reshape(B, N, T, D)

        grid_col = grid.transpose(1, 2).reshape(B * T, N, D)
        grid_col = self.col_attn(grid_col)
        grid = grid_col.reshape(B, T, N, D).transpose(1, 2)

        return grid


class TargetCrossBlock(nn.Module):
    """Second-half block: time attention keeps the model-wide 2x FFN (the same with FULL2D, baseline and profile enhanced model), while
    one-way target-to-aux sequence attention uses a 4x FFN specifically.
    """

    def __init__(self, d_model=64, n_heads=8, dropout=0.2):
        super().__init__()
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        # Column-wise Cross-Attention
        self.cross_attn = CrossSequenceBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)

    def forward(self, target_feat, aux_feat):
        # target_feat: [B, 1, 192, d_model]
        # aux_feat: [B, 10, 192, d_model]
        B, _, T, D = target_feat.shape
        K = aux_feat.shape[1]  # K=10

        target_time = target_feat.squeeze(1)  # [B, 192, D]
        target_time = self.time_attn(target_time)
        target_feat = target_time.unsqueeze(1)  # [B, 1, 192, D]

        q = target_feat.transpose(1, 2).reshape(B * T, 1, D)
        kv = aux_feat.transpose(1, 2).reshape(B * T, K, D)

        q_out = self.cross_attn(q, kv)
        target_feat = q_out.reshape(B, T, 1, D).transpose(1, 2)

        return target_feat


class ShortTermPredictorWithFuture(nn.Module):
    """Evoformer-based heavy predictor."""

    def __init__(
            self,
            seq_len_short=144,
            pred_len=48,
            in_channels=10,
            met_channels=9,
            d_profile=256,
            d_model=64,
            n_heads=8,
            e_layers=6,
            dropout=0.2,
    ):
        super(ShortTermPredictorWithFuture, self).__init__()
        self.seq_len_short = seq_len_short
        self.pred_len = pred_len
        self.total_len = seq_len_short + pred_len
        self.d_model = d_model

        # 1D Convolution for feature extraction
        self.feature_conv = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # Positional Encoding
        self.time_pe = nn.Parameter(torch.randn(1, 1, self.total_len, d_model) * 0.02)

        # Separation embeddings
        self.target_pe = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
        self.aux_pe = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)

        self.profile_proj = nn.Sequential(
            nn.Linear(d_profile, d_model),
            nn.GELU()
        )

        # Evoformer blocks
        half_layers = e_layers // 2

        self.full_blocks = nn.ModuleList([
            Full2DBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(half_layers)
        ])

        # Aux reconstruction head
        self.aux_reconstruct_head = nn.Linear(d_model, 1)

        self.target_blocks = nn.ModuleList([
            TargetCrossBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(e_layers - half_layers)
        ])

        # Output head
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x_short, profile, future_met, aux_profiles=None, mask=None, matched_hist=None):
        B = x_short.shape[0]

        last_pm25 = x_short[:, -1, -1:]
        pad_pm25 = last_pm25.unsqueeze(1).expand(B, self.pred_len, 1)
        future_x = torch.cat([future_met, pad_pm25], dim=-1)
        target_seq = torch.cat([x_short, future_x], dim=1).unsqueeze(1)

        aux_seqs = matched_hist
        grid = torch.cat([target_seq, aux_seqs], dim=1)
        N_seq = grid.shape[1]

        grid_flat = grid.reshape(B * N_seq, self.total_len, -1).transpose(1, 2)
        grid_conv = self.feature_conv(grid_flat)
        grid = grid_conv.transpose(1, 2).reshape(B, N_seq, self.total_len, self.d_model)

        # Add PEs
        grid = grid + self.time_pe
        grid[:, 0:1, :, :] = grid[:, 0:1, :, :] + self.target_pe
        grid[:, 1:, :, :] = grid[:, 1:, :, :] + self.aux_pe

        # Inject target profile
        prof_emb = self.profile_proj(profile).reshape(B, 1, 1, self.d_model)
        grid[:, 0:1, :, :] += prof_emb

        # Inject aux profiles
        if aux_profiles is not None:
            # aux_profiles: [B, 10, d_profile]
            aux_prof_emb = self.profile_proj(aux_profiles).unsqueeze(2)  # [B, 10, 1, d_model]
            grid[:, 1:, :, :] += aux_prof_emb

        # Full 2D updates
        for block in self.full_blocks:
            grid = block(grid)

        target_feat = grid[:, 0:1, :, :]
        aux_feat = grid[:, 1:, :, :]

        # Aux sequence reconstruction
        aux_reconstruct = self.aux_reconstruct_head(aux_feat).squeeze(-1)  # [B, 10, 192]

        # Cross-Attention updates
        for block in self.target_blocks:
            target_feat = block(target_feat, aux_feat)

        target_future = target_feat[:, 0, -self.pred_len:, :]
        delta = self.out_proj(target_future).squeeze(-1)
        pm25_pred = delta + last_pm25

        return pm25_pred, aux_reconstruct

class CNN1DEncoder(nn.Module):
    """1D-CNN lightweight encoder."""
    def __init__(self, in_channels=10, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128, d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        feat = self.net(x)
        return self.fc(feat)


class DataSet():
    def __init__(self, path):
        raw_data = np.load(path)
        raw_data = np.transpose(raw_data, (0, 2, 1))
        raw_data = raw_data[:, -365 * 24 * 4:, :]

        self.raw_data = raw_data
        self.train_end = 365 * 24 * 2
        self.val_end = 365 * 24 * 3

        (self.met_data_normalized,
         self.pol_data_normalized,
         self.pol_mask_matrix,
         self.met_mean,
         self.met_std,
         self.pol_mean,
         self.pol_std) = self.normalize_data(self.raw_data)

    def normalize_data(self, raw_data):
        met_data_raw = raw_data[:, :, 0:9]
        pol_data_raw = raw_data[:, :, 9]
        met_data_raw = torch.from_numpy(met_data_raw).float()
        pol_data_raw = torch.from_numpy(pol_data_raw).float()

        calc_start = 0
        calc_end = 365 * 24 * 2

        met_calc_data = met_data_raw[:, calc_start:calc_end, :]
        pol_calc_data = pol_data_raw[:, calc_start:calc_end]
        valid_pm25_calc = (
            ~torch.isnan(pol_calc_data)
        ) & (pol_calc_data <= 1000)

        C = met_data_raw.shape[2]
        met_mean, met_std = [], []
        met_data_normalized = torch.zeros_like(met_data_raw)

        for i in range(C):
            slice_calc = met_calc_data[:, :, i]
            slice_full = met_data_raw[:, :, i]

            mean_i = slice_calc.mean().item()
            std_i = slice_calc.std().item()

            met_mean.append(mean_i)
            met_std.append(std_i)
            met_data_normalized[:, :, i] = (slice_full - mean_i) / (std_i + 1e-8)

        pol_mask_matrix = torch.where(
            torch.isnan(pol_data_raw),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32)
        )

        pol_data_filled = torch.nan_to_num(pol_data_raw, nan=0.0)

        valid_values_calc = pol_calc_data[valid_pm25_calc]
        pol_mean = valid_values_calc.mean()
        pol_std = valid_values_calc.std()

        # Normalize PM2.5 globally
        pol_data_normalized = (pol_data_filled - pol_mean) / (pol_std + 1e-8)
        pol_data_normalized = pol_data_normalized * pol_mask_matrix

        return (
            met_data_normalized,
            pol_data_normalized,
            pol_mask_matrix,
            met_mean,
            met_std,
            pol_mean.item(),
            pol_std.item()
        )
