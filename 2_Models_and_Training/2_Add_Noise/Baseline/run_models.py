import os

import random
import torch

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tslearn.metrics import SoftDTWLossPyTorch
from mult_model import BaselineTransformer, DataSet
import sys



class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# Dataset
class PredictionDatasetWithFuture(Dataset):
    """Dataset backed by precomputed quality-filtered windows."""

    def __init__(self, met_data, pol_data, mask_data, cache_file,
                 T_short=144, pred_len=48,
                 R_stations=32, num_iterations=500):
        self.T_short = T_short
        self.pred_len = pred_len
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

        print(f"Loaded cache: {cache_file} (Eligible stations: {len(self.eligible_stations)}/{self.N_stations})")

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, index):
        perm = torch.randperm(len(self.eligible_stations))[:self.R]
        station_ids = self.eligible_stations[perm]

        station_ids_list = station_ids.tolist()
        starts = []
        for sid in station_ids_list:
            vs = self.valid_starts_per_station[sid]
            idx = random.randint(0, len(vs) - 1)
            starts.append(vs[idx].item())

        device = self.x_data.device
        starts_tensor = torch.tensor(starts, dtype=torch.long, device=device)
        station_ids_dev = station_ids.to(device)

        sid_idx = station_ids_dev.unsqueeze(1)
        short_idx = starts_tensor.unsqueeze(1) + torch.arange(self.T_short, device=device)
        target_idx = (starts_tensor + self.T_short).unsqueeze(1) + torch.arange(self.pred_len, device=device)

        short_segs = self.x_data[sid_idx, short_idx, :]
        short_masks = self.mask_data[sid_idx, short_idx]
        targets = self.x_data[sid_idx, target_idx, :]
        future_mets = targets[:, :, :-1]

        return short_segs, short_masks, future_mets, targets, station_ids


# Loss
def pm25_loss(preds, targets, pol_mean, pol_std, gamma=0.25, k_ratio=0.1):
    """Combined loss with Dynamic Sequence Scaling."""
    pm25_targets = targets[:, :, -1]
    pm25_targets_raw = pm25_targets * pol_std + pol_mean
    valid_mask = (pm25_targets_raw <= 1000).all(dim=1)

    preds_filtered = preds[valid_mask]
    targets_filtered = pm25_targets[valid_mask]

    # Return 0 if no valid data
    if preds_filtered.shape[0] == 0:
        zero_val = preds.sum() * 0.0
        return (zero_val, zero_val.detach(), zero_val.detach(),
                zero_val.detach(), zero_val.detach())

    # 1. Base Loss
    abs_diff_base = torch.abs(preds_filtered - targets_filtered)
    loss_base_seq = abs_diff_base.mean(dim=1)  # Shape: (N,)

    # 2. Trend Loss
    diff_preds = preds_filtered[:, 1:] - preds_filtered[:, :-1]
    diff_targets = targets_filtered[:, 1:] - targets_filtered[:, :-1]
    loss_trend_seq = torch.abs(diff_preds - diff_targets).mean(dim=1)  # Shape: (N,)

    # 3. Soft-DTW Loss
    sdtw_criterion = SoftDTWLossPyTorch(gamma=gamma)
    preds_3d = preds_filtered.unsqueeze(2)
    targets_3d = targets_filtered.unsqueeze(2)

    dtw_xy = sdtw_criterion(preds_3d, targets_3d)
    dtw_xx = sdtw_criterion(preds_3d, preds_3d)
    with torch.no_grad():
        dtw_yy = sdtw_criterion(targets_3d, targets_3d)

    loss_divergence_seq = (
        (dtw_xy - 0.5 * (dtw_xx + dtw_yy)).clamp(min=0.0) /
        targets_filtered.shape[1]
    )

    # 4. Top-K Loss
    seq_len = targets_filtered.shape[1]
    k_num = max(1, int(seq_len * k_ratio))

    topk_target_vals, topk_indices = torch.topk(targets_filtered, k=k_num, dim=1)
    topk_pred_vals = torch.gather(preds_filtered, 1, topk_indices)

    abs_diff_topk = torch.abs(topk_pred_vals - topk_target_vals)
    loss_topk_seq = abs_diff_topk.mean(dim=1)  # Shape: (N,)

    # 5. Total Weighted Loss
    alpha = 1.0  # Base 权重
    beta = 0.5  # Trend 权重
    gamma_w = 2.4  # DTW 权重
    delta_w = 0.3  # Top-K 权重

    total_loss_seq = (alpha * loss_base_seq) + \
                     (beta * loss_trend_seq) + \
                     (gamma_w * loss_divergence_seq) + \
                     (delta_w * loss_topk_seq)

    # 6. Dynamic Scaling
    limit = 2.0
    scale_factors = torch.where(
        total_loss_seq > limit,
        limit / total_loss_seq.detach(),
        torch.ones_like(total_loss_seq)
    )

    final_loss = (total_loss_seq * scale_factors).mean()

    scaled_base = (loss_base_seq * scale_factors).mean().detach()
    scaled_trend = (loss_trend_seq * scale_factors).mean().detach()
    scaled_dtw = (loss_divergence_seq * scale_factors).mean().detach()
    scaled_topk = (loss_topk_seq * scale_factors).mean().detach()
    return final_loss, scaled_base, scaled_trend, scaled_dtw, scaled_topk


def pm25_no_trend_loss_components(preds, targets, pol_mean, pol_std,
                                  gamma=0.25, k_ratio=0.1):
    pm25_targets = targets[:, :, -1]
    pm25_targets_raw = pm25_targets * pol_std + pol_mean
    valid_mask = (pm25_targets_raw <= 1000).all(dim=1)

    preds = preds[valid_mask]
    pm25_targets = pm25_targets[valid_mask]

    loss_base_seq = torch.abs(preds - pm25_targets).mean(dim=1)

    sdtw_criterion = SoftDTWLossPyTorch(gamma=gamma)
    preds_3d = preds.unsqueeze(2)
    targets_3d = pm25_targets.unsqueeze(2)
    dtw_xy = sdtw_criterion(preds_3d, targets_3d)
    dtw_xx = sdtw_criterion(preds_3d, preds_3d)
    dtw_yy = sdtw_criterion(targets_3d, targets_3d)
    loss_dtw_seq = (
        (dtw_xy - 0.5 * (dtw_xx + dtw_yy)).clamp(min=0.0) /
        pm25_targets.shape[1]
    )

    k_num = max(1, int(pm25_targets.shape[1] * k_ratio))
    topk_target_vals, topk_indices = torch.topk(pm25_targets, k=k_num, dim=1)
    topk_pred_vals = torch.gather(preds, 1, topk_indices)
    loss_topk_seq = torch.abs(topk_pred_vals - topk_target_vals).mean(dim=1)

    total_loss_seq = loss_base_seq + 2.4 * loss_dtw_seq + 0.3 * loss_topk_seq
    return loss_base_seq, loss_dtw_seq, loss_topk_seq, total_loss_seq


def pm25_validation_loss(preds, targets, pol_mean, pol_std, limit=2.0):
    loss_base_seq, loss_dtw_seq, loss_topk_seq, total_loss_seq = (
        pm25_no_trend_loss_components(
        preds, targets, pol_mean, pol_std
        )
    )
    scale_factors = torch.where(
        total_loss_seq > limit,
        limit / total_loss_seq,
        torch.ones_like(total_loss_seq)
    )
    scaled_base = (loss_base_seq * scale_factors).mean()
    scaled_dtw = (loss_dtw_seq * scale_factors).mean()
    scaled_topk = (loss_topk_seq * scale_factors).mean()
    scaled_total = (total_loss_seq * scale_factors).mean()
    return scaled_base, scaled_dtw, scaled_topk, scaled_total


def build_model(device, in_channels=10, T_short=144, pred_len=48, nhead=8):
    return BaselineTransformer(
        in_channels=in_channels,
        d_model=256,
        nhead=nhead,
        max_len=T_short + pred_len,
        num_layers=4,
        dropout=0.1
    ).to(device)


def run_experiment(exp_dir, cache_dir, data_path, seed, nhead=8):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    # Hyperparameters
    T_short = 144
    pred_len = 48
    R_stations = 32
    in_channels = 10
    epochs = 100
    max_lr = 2e-4
    warmup_start_lr = 1e-5
    warmup_epochs = 1
    scheduler_t_max = 100

    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger(os.path.join(
        exp_dir, f"main_model_training_seed_{seed}.log"
    ))
    sys.stdout = logger
    print(f"Starting Transformer experiment with seed {seed} in {exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_set = DataSet(data_path)
    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    pol_mean = data_set.pol_mean
    pol_std = data_set.pol_std

    one_year_steps = 365 * 24
    train_end = one_year_steps * 2
    val_end = one_year_steps * 3

    train_met = met_data[:, :train_end, :].to(device)
    train_pol = pol_data[:, :train_end].to(device)
    train_mask = mask_data[:, :train_end].to(device)

    val_met = met_data[:, train_end:val_end, :].to(device)
    val_pol = pol_data[:, train_end:val_end].to(device)
    val_mask = mask_data[:, train_end:val_end].to(device)

    train_cache_file = os.path.join(
        cache_dir,
        f"dataset_cache_train_2y_T{T_short}_P{pred_len}_seed_{seed}.pt"
    )
    val_cache_file = os.path.join(
        cache_dir,
        f"dataset_cache_val_1y_T{T_short}_P{pred_len}_seed_{seed}.pt"
    )

    train_pred_ds = PredictionDatasetWithFuture(
        train_met, train_pol, train_mask,
        cache_file=train_cache_file,
        T_short=T_short, pred_len=pred_len,
        R_stations=R_stations, num_iterations=1500
    )

    val_pred_ds = PredictionDatasetWithFuture(
        val_met, val_pol, val_mask,
        cache_file=val_cache_file,
        T_short=T_short, pred_len=pred_len,
        R_stations=R_stations, num_iterations=600
    )

    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    train_pred_loader = DataLoader(
        train_pred_ds, batch_size=1, shuffle=True, num_workers=0,
        generator=train_generator
    )
    val_pred_loader = DataLoader(
        val_pred_ds, batch_size=1, shuffle=True, num_workers=0,
        generator=val_generator
    )

    predictor = build_model(device, in_channels, T_short, pred_len, nhead)

    all_params = list(predictor.parameters())
    optimizer = optim.AdamW(all_params, lr=max_lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max, eta_min=1e-6
    )

    ema_alpha = 0.25
    ema_no_trend_loss = None
    best_ema_no_trend_loss = float('inf')
    best_model_path = os.path.join(
        exp_dir, f"best_main_model_seed_{seed}.pth"
    )

    # Training loop
    for epoch in range(epochs):
        predictor.train()

        prediction_loss_total = 0.0
        train_base_total = 0.0
        train_trend_total = 0.0
        train_dtw_total = 0.0
        train_topk_total = 0.0

        num_train_batches = max(len(train_pred_loader), 1)

        for batch_idx, (short_segs, short_masks, future_mets, targets, station_ids) in enumerate(train_pred_loader):
            # LR Warmup
            if epoch < warmup_epochs:
                warmup_steps = warmup_epochs * len(train_pred_loader)
                completed_steps = epoch * len(train_pred_loader) + batch_idx
                warmup_progress = completed_steps / max(warmup_steps - 1, 1)
                current_lr = warmup_start_lr + (
                    max_lr - warmup_start_lr
                ) * warmup_progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            short_segs = short_segs.squeeze(0).to(device)
            short_masks = short_masks.squeeze(0).to(device)
            future_mets = future_mets.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)

            optimizer.zero_grad()
            preds = predictor(short_segs, future_met=future_mets, mask=short_masks)

            loss, l_base, l_trend, l_dtw, l_topk = pm25_loss(
                preds, targets, pol_mean=pol_mean, pol_std=pol_std
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            prediction_loss_total += loss.item()
            train_base_total += l_base.item()
            train_trend_total += l_trend.item()
            train_dtw_total += l_dtw.item()
            train_topk_total += l_topk.item()

        scheduler.step()

        num_train_batches = max(len(train_pred_loader), 1)
        avg_prediction_loss = prediction_loss_total / num_train_batches
        avg_t_base = train_base_total / num_train_batches
        avg_t_trend = train_trend_total / num_train_batches
        avg_t_dtw = train_dtw_total / num_train_batches
        avg_t_topk = train_topk_total / num_train_batches

        # Validation
        predictor.eval()
        val_base_total = 0.0
        val_dtw_total = 0.0
        val_topk_total = 0.0
        val_no_trend_total = 0.0

        with torch.no_grad():
            for short_segs, short_masks, future_mets, targets, station_ids in val_pred_loader:
                short_segs = short_segs.squeeze(0).to(device)
                short_masks = short_masks.squeeze(0).to(device)
                future_mets = future_mets.squeeze(0).to(device)
                targets = targets.squeeze(0).to(device)

                preds = predictor(short_segs, future_met=future_mets, mask=short_masks)

                lv_base, lv_dtw, lv_topk, lv_no_trend = pm25_validation_loss(
                    preds, targets, pol_mean=pol_mean, pol_std=pol_std
                )
                val_base_total += lv_base.item()
                val_dtw_total += lv_dtw.item()
                val_topk_total += lv_topk.item()
                val_no_trend_total += lv_no_trend.item()

        num_val_batches = max(len(val_pred_loader), 1)
        avg_v_base = val_base_total / num_val_batches
        avg_v_dtw = val_dtw_total / num_val_batches
        avg_v_topk = val_topk_total / num_val_batches
        avg_val_no_trend_loss = val_no_trend_total / num_val_batches
        if ema_no_trend_loss is None:
            ema_no_trend_loss = avg_val_no_trend_loss
        else:
            ema_no_trend_loss = (ema_alpha * avg_val_no_trend_loss +
                                 (1.0 - ema_alpha) * ema_no_trend_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {avg_prediction_loss:.4f} "
              f"(Base: {avg_t_base:.4f}, Trend: {avg_t_trend:.4f}, "
              f"DTW: {avg_t_dtw:.4f}, TopK: {avg_t_topk:.4f}) | "
              f"Val Loss: {avg_val_no_trend_loss:.6f} "
              f"(Base: {avg_v_base:.4f}, DTW: {avg_v_dtw:.4f}, "
              f"TopK: {avg_v_topk:.4f}) | "
              f"EMA No-Trend Loss: {ema_no_trend_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if ema_no_trend_loss < best_ema_no_trend_loss:
            torch.save(predictor.state_dict(), best_model_path)
            best_ema_no_trend_loss = ema_no_trend_loss

    logger.close()
    return best_model_path
