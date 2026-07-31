import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

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


class ShortTermPredictorWithFuture(nn.Module):
    def __init__(
            self,
            seq_history_len=144,
            pred_len=48,
            in_channels=10,
            d_model=128,
            d_profile=256,
            n_heads=8,
            e_layers=6,
            **kwargs
    ):
        super(ShortTermPredictorWithFuture, self).__init__()
        sequence_length = seq_history_len + pred_len

        self.mlp_in = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.profile_proj = nn.Sequential(
            nn.Linear(d_profile, d_model),
            nn.GELU()
        )

        self.time_embedding = nn.Parameter(
            torch.empty(1, sequence_length, d_model)
        )
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dim_feedforward=d_model * 2,
            dropout=kwargs.get('dropout', 0.1)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.mlp_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_short, profile, future_met, mask=None):
        pred_len = future_met.shape[1]

        last_pm25 = x_short[:, -1, -1:]  # (B, 1)
        future_pm25_padded = last_pm25.unsqueeze(1).expand(-1, pred_len, -1)  # (B, pred_len, 1)
        future_x = torch.cat([future_met, future_pm25_padded], dim=-1)  # (B, pred_len, 10)
        full_x = torch.cat([x_short, future_x], dim=1).contiguous()

        x_emb = self.mlp_in(full_x)  # (B, 192, 64)

        # Inject static features
        p_emb = self.profile_proj(profile).unsqueeze(1) # (B, 1, 64)
        x_emb = x_emb + p_emb
        x_emb = (x_emb + self.time_embedding).contiguous()

        out_enc = self.transformer(x_emb)  # (B, 192, 64)
        out_pred = out_enc[:, -pred_len:, :]  # (B, 48, 64)

        delta = self.mlp_out(out_pred).squeeze(-1)  # (B, 48)
        pm25_pred = delta + last_pm25.squeeze(-1).unsqueeze(-1)  # (B, pred_len)

        return pm25_pred


class DataSet():
    def __init__(self, path):
        raw_data = np.load(path)
        raw_data = np.transpose(raw_data, (0, 2, 1))
        raw_data = raw_data[:, -365 * 24 * 4:, :]

        self.raw_data = raw_data

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
            # Apply statistics fitted only on the two-year training range.
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
