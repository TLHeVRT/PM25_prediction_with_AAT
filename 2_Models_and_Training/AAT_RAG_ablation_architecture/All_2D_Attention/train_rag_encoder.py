import os
import torch
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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


def weighted_mae_loss(
        preds,
        targets,
        pol_mean,
        pol_std,
        pm25_weight=5.0,
):
    """Quality-filtered encoder loss; no limit scaling is applied."""
    pm25_targets = targets[..., -1]
    pm25_targets_raw = pm25_targets * pol_std + pol_mean
    valid_mask = (pm25_targets_raw <= 1000.0).all(dim=1)

    preds_filtered = preds[valid_mask]
    targets_filtered = targets[valid_mask]
    if preds_filtered.shape[0] == 0:
        zero_val = preds.sum() * 0.0
        return zero_val, zero_val.detach(), zero_val.detach()

    met_preds = preds_filtered[..., :-1]
    pm25_preds = preds_filtered[..., -1]
    met_targets = targets_filtered[..., :-1]
    pm25_targets = targets_filtered[..., -1]

    loss_met = F.l1_loss(met_preds, met_targets)
    loss_pm25 = F.l1_loss(pm25_preds, pm25_targets)
    total_loss = loss_met + pm25_weight * loss_pm25
    return total_loss, loss_met, loss_pm25


def train_rag_encoder(
        output_dir,
        cache_paths,
        data_set=None,
        T_short=144,
        cache_pred_len=48,
        target_pred_len=6,
        R_stations=128,
        epochs=100,
        lr=5e-5,
        pm25_weight=5.0,
        ema_alpha=0.25,
        seed=None,
):
    """Train one run's RAG encoder on years 1-2 and validate on year 3."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RAG encoder device: {device}")

    if data_set is None:
        print("Loading raw data for RAG encoder...")
        data_set = DataSet('data_matrix.npy')
    met_data = data_set.met_data_normalized.to(device)
    pol_data = data_set.pol_data_normalized.to(device)
    mask_data = data_set.pol_mask_matrix.to(device)

    train_end = data_set.train_end
    val_end = data_set.val_end

    train_met = met_data[:, :train_end, :]
    train_pol = pol_data[:, :train_end]
    train_mask = mask_data[:, :train_end]
    val_met = met_data[:, train_end:val_end, :]
    val_pol = pol_data[:, train_end:val_end]
    val_mask = mask_data[:, train_end:val_end]

    train_ds = PredictionDatasetWithCache(train_met, train_pol, train_mask, cache_paths["train"],
                                          T_short=T_short, target_pred_len=target_pred_len,
                                          R_stations=R_stations, num_iterations=1000)
    val_ds = PredictionDatasetWithCache(val_met, val_pol, val_mask, cache_paths["val"],
                                        T_short=T_short, target_pred_len=target_pred_len,
                                        R_stations=R_stations, num_iterations=200)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = LightweightSeq2Seq(in_channels=10, d_model=128, pred_len=target_pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ema_total_val_loss = None
    best_ema_total_val_loss = float('inf')
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    save_path = os.path.join(output_dir, f"best_encoder{seed_suffix}.pth")

    print("Starting RAG encoder training (EMA-selected total validation loss)...")
    for epoch in range(epochs):
        # -- Training --
        model.train()
        train_loss_total = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.squeeze(0)  # [R_stations, 144, 10]
            targets = targets.squeeze(0)  # [R_stations, 6, 10]

            optimizer.zero_grad()
            preds = model(inputs)

            loss, _, _ = weighted_mae_loss(
                preds,
                targets,
                pol_mean=data_set.pol_mean,
                pol_std=data_set.pol_std,
                pm25_weight=pm25_weight,
            )
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        model.eval()
        val_loss_total = 0.0
        val_met_total, val_pm25_total = 0.0, 0.0

        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs = inputs.squeeze(0)
                targets = targets.squeeze(0)

                preds = model(inputs)
                loss, l_met, l_pm25 = weighted_mae_loss(
                    preds,
                    targets,
                    pol_mean=data_set.pol_mean,
                    pol_std=data_set.pol_std,
                    pm25_weight=pm25_weight,
                )

                val_loss_total += loss.item()
                val_met_total += l_met.item()
                val_pm25_total += l_pm25.item()

        avg_val_loss = val_loss_total / len(val_loader)
        avg_v_met = val_met_total / len(val_loader)
        avg_v_pm25 = val_pm25_total / len(val_loader)

        if ema_total_val_loss is None:
            ema_total_val_loss = avg_val_loss
        else:
            ema_total_val_loss = (
                ema_alpha * avg_val_loss
                + (1.0 - ema_alpha) * ema_total_val_loss
            )

        print(f"Epoch [{epoch + 1:03d}/{epochs:03d}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Total Loss: {avg_val_loss:.4f} "
              f"(Met MAE: {avg_v_met:.4f}, PM2.5 MAE: {avg_v_pm25:.4f}) | "
              f"EMA Total Loss: {ema_total_val_loss:.6f}")

        if ema_total_val_loss < best_ema_total_val_loss:
            best_ema_total_val_loss = ema_total_val_loss
            torch.save(model.encoder.state_dict(), save_path)
            print(
                f"Saved best RAG encoder to {save_path} "
                f"(EMA total loss: {ema_total_val_loss:.6f})."
            )

    del model, optimizer, train_ds, val_ds, train_loader, val_loader
    del met_data, pol_data, mask_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_path


def main():
    raise SystemExit(
        "Run the complete seeded workflow with "
        "`python run_parallel_pipeline.py --seeds SEED [SEED ...]`."
    )


if __name__ == '__main__':
    main()
