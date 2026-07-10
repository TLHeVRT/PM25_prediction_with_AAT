import math

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd

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


class PositionalEncoding(nn.Module):
    """Standard Transformer Positional Encoding."""

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]


class ShortTermPredictorWithFuture(nn.Module):
    def __init__(
            self,
            in_channels=10,
            d_model=64,
            d_profile=256,
            n_heads=4,
            e_layers=3,
            max_len=200,
            **kwargs
    ):
        super(ShortTermPredictorWithFuture, self).__init__()

        self.mlp_in = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.profile_proj = nn.Sequential(
            nn.Linear(d_profile, d_model),
            nn.GELU()
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

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

        self.mlp_out = nn.Linear(d_model, 1)

    def forward(self, x_short, profile, future_met, mask=None):
        B = x_short.shape[0]
        pred_len = future_met.shape[1]

        last_pm25 = x_short[:, -1, -1:]  # (B, 1)
        future_pm25_padded = last_pm25.unsqueeze(1).expand(-1, pred_len, -1)  # (B, pred_len, 1)
        future_x = torch.cat([future_met, future_pm25_padded], dim=-1)  # (B, pred_len, 10)
        full_x = torch.cat([x_short, future_x], dim=1).contiguous()

        x_emb = self.mlp_in(full_x)  # (B, 192, 64)

        # Inject static features
        p_emb = self.profile_proj(profile).unsqueeze(1) # (B, 1, 64)
        x_emb = x_emb + p_emb
        x_emb = self.pos_encoder(x_emb).contiguous()

        out_enc = self.transformer(x_emb)  # (B, 192, 64)
        out_pred = out_enc[:, -pred_len:, :]  # (B, 48, 64)

        delta = self.mlp_out(out_pred).squeeze(-1)  # (B, 48)
        pm25_pred = delta + last_pm25.squeeze(-1).unsqueeze(-1)  # (B, pred_len)

        return pm25_pred


class DataSet():
    def __init__(self, path, diagnosis_path="diagnosis_output/02_sensor_stuck_detection.csv"):
        raw_data = np.load(path)
        raw_data = np.transpose(raw_data, (0, 2, 1))
        raw_data = raw_data[:, -365 * 24 * 4:, :]
        self.valid_stations = None

        if diagnosis_path and os.path.exists(diagnosis_path):
            df_stuck = pd.read_csv(diagnosis_path)
            valid_stations = df_stuck[df_stuck['pm25_max_run'] < 12]['station_id'].values
            self.valid_stations = valid_stations

            raw_data = raw_data[valid_stations, :, :]

        self.raw_data = raw_data

        (self.met_data_normalized,
         self.pol_data_normalized,
         self.pol_mask_matrix,
         self.met_mean,
         self.met_std,
         self.pol_mean,
         self.pol_std) = self.normalize_data(self.raw_data)

        pm25_raw = raw_data[:, :, 9]
        max_val = np.nanmax(pm25_raw)
        abnormal_count = np.sum(pm25_raw > 1000)

    def normalize_data(self, raw_data):
        met_data_raw = raw_data[:, :, 0:9]
        pol_data_raw = raw_data[:, :, 9]
        met_data_raw = torch.from_numpy(met_data_raw).float()

        calc_start = 365 * 24 * 1
        calc_end = 365 * 24 * 3

        met_calc_data = met_data_raw[:, calc_start:calc_end, :]

        N, T, C = met_data_raw.shape
        met_mean, met_std = [], []
        met_data_normalized = torch.zeros_like(met_data_raw)

        for i in range(C):
            slice_calc = met_calc_data[:, :, i]
            slice_full = met_data_raw[:, :, i]

            mean_i = slice_calc.mean().item()
            std_i = slice_calc.std().item()

            met_mean.append(mean_i)
            met_std.append(std_i)
            # Normalize globally using year 2-3 stats
            met_data_normalized[:, :, i] = (slice_full - mean_i) / (std_i + 1e-8)

        pol_data_raw = torch.from_numpy(pol_data_raw).float()

        pol_mask_matrix = torch.where(
            torch.isnan(pol_data_raw),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32)
        )

        pol_data_filled = torch.nan_to_num(pol_data_raw, nan=0.0)

        pol_calc_mask = pol_mask_matrix[:, calc_start:calc_end].bool()
        pol_calc_filled = pol_data_filled[:, calc_start:calc_end]
        valid_values_calc = pol_calc_filled[pol_calc_mask]

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