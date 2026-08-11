import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BaselineTransformer(nn.Module):
    ### Baseline Transformer used in the final submission.

    def __init__(self, in_channels=10, d_model=64, nhead=8, num_layers=3,
                 max_len=200, dropout=0.1):
        super(BaselineTransformer, self).__init__()

        self.mlp_in = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = dropout
        # 时间维度上的 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dim_feedforward=d_model * 2,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )


    def forward(self, x_short, future_met, mask=None):
        B = x_short.shape[0]
        pred_len = future_met.shape[1]

        last_pm25 = x_short[:, -1, -1:]  # (B, 1)

        future_pm25_padded = last_pm25.unsqueeze(1).expand(-1, pred_len, -1)  # (B, pred_len, 1)

        future_x = torch.cat([future_met, future_pm25_padded], dim=-1)  # (B, pred_len, 10)

        full_x = torch.cat([x_short, future_x], dim=1).contiguous()

        x_emb = self.mlp_in(full_x)  # (B, 192, 64)
        x_emb = self.pos_encoder(x_emb).contiguous()

        out_enc = self.transformer(x_emb)  # (B, 192, 64)

        out_pred = out_enc[:, -pred_len:, :]  # (B, 48, 64)

        # 7. 残差映射
        delta = self.mlp_out(out_pred).squeeze(-1)  # (B, 48)
        pm25_pred = delta + last_pm25.squeeze(-1).unsqueeze(-1)  # (B, pred_len)

        return pm25_pred

class DataSet():

    def __init__(self, path):
        raw_data = np.load(path)
        raw_data = np.transpose(raw_data, (0, 2, 1))
        # Keep the four most recent years in chronological order.
        raw_data = raw_data[:, -365 * 24 * 4:, :]

        print(f"Data shape: {raw_data.shape}")
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
        print(f"Max PM2.5 value: {max_val}")

        abnormal_count = np.sum(pm25_raw > 1000)
        print(f"Abnormal points (>1000): {abnormal_count}")

    def normalize_data(self, raw_data):
        met_data_raw = raw_data[:, :, 0:9]
        pol_data_raw = raw_data[:, :, 9]
        met_data_raw = torch.from_numpy(met_data_raw).float()

        calc_start = 0
        # Fit normalization on the fourth- and third-most-recent years.
        calc_end = 365 * 24 * 2
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
            # Attn : 使用训练集的均值和方差，对全局数据进行归一化
            met_data_normalized[:, :, i] = (slice_full - mean_i) / (std_i + 1e-8)

        pol_data_raw = torch.from_numpy(pol_data_raw).float()

        pol_mask_matrix = torch.where(
            torch.isnan(pol_data_raw),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32)
        )

        pol_data_filled = torch.nan_to_num(pol_data_raw, nan=0.0)


        pol_calc_raw = pol_data_raw[:, calc_start:calc_end]
        pol_calc_mask = (pol_mask_matrix[:, calc_start:calc_end].bool() &
                         (pol_calc_raw <= 1000))
        pol_calc_filled = pol_data_filled[:, calc_start:calc_end]
        valid_values_calc = pol_calc_filled[pol_calc_mask]

        pol_mean = valid_values_calc.mean()
        pol_std = valid_values_calc.std()

        # Attn : 使用算出的 PM2.5 均值和方差，对全局数据进行归一化
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
