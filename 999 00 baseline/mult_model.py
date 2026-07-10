import numpy as np
import torch
import math
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd


class BaselineLSTM(nn.Module):
    ### Worse than baseline Transformer, abandoned

    def __init__(self, in_channels=10, met_channels=9, hidden_size=64, num_layers=2):
        super(BaselineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.hist_lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dec_cells = nn.ModuleList([
            nn.LSTMCell(1 + met_channels if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

        self.predictor = nn.Linear(hidden_size, 1)

    def forward(self, x_short, future_met, mask=None):
        B = x_short.size(0)
        pred_len = future_met.size(1)

        _, (hx, cx) = self.hist_lstm(x_short)  # hx, cx shape: (num_layers, B, hidden_size)

        hx_list = [hx[i] for i in range(self.num_layers)]
        cx_list = [cx[i] for i in range(self.num_layers)]

        last_pm25 = x_short[:, -1, -1:]  # (B, 1)
        curr_pm25 = last_pm25

        preds = []

        for t in range(pred_len):
            curr_met = future_met[:, t, :]  # (B, 9)
            dec_in = torch.cat([curr_pm25, curr_met], dim=-1)  # (B, 10)

            for i in range(self.num_layers):
                hx_list[i], cx_list[i] = self.dec_cells[i](dec_in, (hx_list[i], cx_list[i]))
                dec_in = hx_list[i]  

            curr_pm25 = self.predictor(hx_list[-1])  # (B, 1)
            preds.append(curr_pm25)

        preds = torch.cat(preds, dim=1)  # (B, pred_len)
        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BaselineTransformer(nn.Module):
    ### baseline we used in the final submission, better than LSTM. iTransformer is also tried before, but it is not better than this simple Transformer on our task.

    def __init__(self, in_channels=10, d_model=64, nhead=4, num_layers=3, max_len=200,dropout=0.1):
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

        self.mlp_out = nn.Linear(d_model, 1)


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

    def __init__(self, path, diagnosis_path="diagnosis_output/02_sensor_stuck_detection.csv"):
        raw_data = np.load(path)
        raw_data = np.transpose(raw_data, (0, 2, 1))
        raw_data = raw_data[:, -365 * 24 * 4:, :]
        self.valid_stations = None  # 记录有效站点的索引

        if diagnosis_path and os.path.exists(diagnosis_path):
            df_stuck = pd.read_csv(diagnosis_path)
            valid_stations = df_stuck[df_stuck['pm25_max_run'] < 12]['station_id'].values
            self.valid_stations = valid_stations

            raw_data = raw_data[valid_stations, :, :]
            print(f"Filtered stations based on diagnosis. Remaining: {len(valid_stations)}")
        else:
            print(f"Diagnosis file not found. Using all {raw_data.shape[0]} stations.")

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
            # Attn : 使用第二和第三年算出的均值和方差，对全局数据进行归一化
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
