import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mult_model import DataSet, LightweightSeq2Seq

class PredictionDatasetWithCache(Dataset):
    """Dataset for RAG encoder training (target 6 hours)."""

    def __init__(self, met_data, pol_data, mask_data, cache_file,
                 T_short=144, target_pred_len=6, R_stations=32, num_iterations=500):
        self.T_short = T_short
        self.target_pred_len = target_pred_len
        self.R = R_stations
        self.num_iterations = num_iterations

        self.x_data = torch.cat([met_data, pol_data.unsqueeze(-1)], dim=-1)
        self.mask_data = mask_data
        self.N_stations = met_data.shape[0]

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        cache_data = torch.load(cache_file)
        self.valid_starts_per_station = cache_data['valid_starts']
        self.eligible_stations = cache_data['eligible_stations']

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, index):
        perm = torch.randperm(len(self.eligible_stations))[:self.R]
        station_ids = self.eligible_stations[perm]

        starts = []
        for sid in station_ids.tolist():
            vs = self.valid_starts_per_station[sid]
            idx = random.randint(0, len(vs) - 1)
            starts.append(vs[idx].item())

        device = self.x_data.device
        starts_tensor = torch.tensor(starts, dtype=torch.long, device=device)
        station_ids_dev = station_ids.to(device)

        sid_idx = station_ids_dev.unsqueeze(1)

        short_idx = starts_tensor.unsqueeze(1) + torch.arange(self.T_short, device=device)
        target_idx = (starts_tensor + self.T_short).unsqueeze(1) + torch.arange(self.target_pred_len, device=device)

        inputs = self.x_data[sid_idx, short_idx, :]  # [R, 144, 10]
        targets = self.x_data[sid_idx, target_idx, :]  # [R, 6, 10]

        return inputs, targets


def weighted_mae_loss(preds, targets, pm25_weight=5.0):
    """Weighted MAE loss for PM2.5."""
    met_preds, pm25_preds = preds[..., :-1], preds[..., -1]
    met_targets, pm25_targets = targets[..., :-1], targets[..., -1]

    loss_met = F.l1_loss(met_preds, met_targets)
    loss_pm25 = F.l1_loss(pm25_preds, pm25_targets)

    total_loss = loss_met + pm25_weight * loss_pm25
    return total_loss, loss_met, loss_pm25


def main():
    # Configurations
    T_short = 144
    cache_pred_len = 48
    target_pred_len = 6
    R_stations = 128
    epochs = 50
    lr = 5e-5
    pm25_weight = 5.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading raw data...")
    data_set = DataSet('data_matrix.npy')
    met_data = data_set.met_data_normalized.to(device)
    pol_data = data_set.pol_data_normalized.to(device)
    mask_data = data_set.pol_mask_matrix.to(device)

    total_time_steps = met_data.shape[1]
    split_idx = int(total_time_steps * (3 / 4))

    train_met, train_pol, train_mask = met_data[:, :split_idx, :], pol_data[:, :split_idx], mask_data[:, :split_idx]
    val_met, val_pol, val_mask = met_data[:, split_idx:, :], pol_data[:, split_idx:], mask_data[:, split_idx:]

    train_cache_file = f"dataset_cache_train_T{T_short}_P{cache_pred_len}.pt"
    val_cache_file = f"dataset_cache_val_T{T_short}_P{cache_pred_len}.pt"

    train_ds = PredictionDatasetWithCache(train_met, train_pol, train_mask, train_cache_file,
                                          T_short=T_short, target_pred_len=target_pred_len,
                                          R_stations=R_stations, num_iterations=1000)
    val_ds = PredictionDatasetWithCache(val_met, val_pol, val_mask, val_cache_file,
                                        T_short=T_short, target_pred_len=target_pred_len,
                                        R_stations=R_stations, num_iterations=200)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = LightweightSeq2Seq(in_channels=10, d_model=128, pred_len=target_pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    save_path = "best_encoder_only.pth"

    print("Starting training (saving Best Encoder only)...")
    for epoch in range(epochs):
        # -- Training --
        model.train()
        train_loss_total = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.squeeze(0)  # [R_stations, 144, 10]
            targets = targets.squeeze(0)  # [R_stations, 6, 10]

            optimizer.zero_grad()
            preds = model(inputs)

            loss, _, _ = weighted_mae_loss(preds, targets, pm25_weight=pm25_weight)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        model.eval()
        val_loss_total = 0.0
        val_met_total, val_pm25_total = 0.0, 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.squeeze(0)
                targets = targets.squeeze(0)

                preds = model(inputs)
                loss, l_met, l_pm25 = weighted_mae_loss(preds, targets, pm25_weight=pm25_weight)

                val_loss_total += loss.item()
                val_met_total += l_met.item()
                val_pm25_total += l_pm25.item()

        avg_val_loss = val_loss_total / len(val_loader)
        avg_v_met = val_met_total / len(val_loader)
        avg_v_pm25 = val_pm25_total / len(val_loader)

        print(f"Epoch [{epoch + 1:03d}/{epochs:03d}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} (Met MAE: {avg_v_met:.4f}, PM2.5 MAE: {avg_v_pm25:.4f})")

        # Save Encoder weights if val loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Val loss improved. Saved encoder weights to {save_path}")
            torch.save(model.encoder.state_dict(), save_path)


if __name__ == '__main__':
    main()