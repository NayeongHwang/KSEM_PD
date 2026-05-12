"""
KSEM Particle Detector 1D-CNN Autoencoder
==========================================
데이터: 12개 CSV (날짜 4일 × PD 3개)
구조: 1440 rows(분 단위) × 256 channels (A0~A127, B0~B127)
목표: 에너지 스펙트럼 압축 + 이벤트 탐지
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────
class Config:
    DATA_DIR    = Path("__file__").parent          # CSV 파일 위치
    LATENT_DIM  = 32                 # 압축 차원 (16~64 사이 권장)
    BATCH_SIZE  = 64
    EPOCHS      = 50
    LR          = 1e-3
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    # 학습: Quiet 기간 (5/10~11), 평가: 전체 (5/10~13)
    TRAIN_DATES = ["20240510", "20240511"]
    ALL_DATES   = ["20240510", "20240511", "20240512", "20240513"]
    TRASH_COLS  = ["A127(Trash)", "B127(Trash)"]


cfg = Config()
print(f"Device: {cfg.DEVICE}")
print(f"Latent dim: {cfg.LATENT_DIM}  →  압축률: {254/cfg.LATENT_DIM:.1f}x")


# ──────────────────────────────────────────────
# 1. 데이터 로딩 & 전처리
# ──────────────────────────────────────────────
def load_pd_data(data_dir: Path, dates: list, pd_ids: list = [1, 2, 3]) -> pd.DataFrame:
    """여러 날짜 × PD를 합쳐서 하나의 DataFrame으로 반환"""
    frames = []
    for date in dates:
        for pd_id in pd_ids:
            fpath = data_dir / f"{date}_PD{pd_id}_Raw Count.csv"
            if not fpath.exists():
                print(f"  [경고] 파일 없음: {fpath.name}")
                continue
            df = pd.read_csv(fpath)
            df["date"]   = date
            df["pd_id"]  = pd_id
            df["Time"]   = pd.to_datetime(df["Time"])
            frames.append(df)
            print(f"  로드: {fpath.name}  →  {df.shape}")
    return pd.concat(frames, ignore_index=True)


def get_spectral_columns(df: pd.DataFrame) -> list:
    """A0~A126, B0~B126 채널 컬럼만 추출 (Trash 제외)"""
    all_cols = [c for c in df.columns if c not in ["Time", "date", "pd_id"]]
    return [c for c in all_cols if c not in cfg.TRASH_COLS]


# 전역 변수로 scaler 저장 (train 기준으로 fit, eval에 동일 적용)
_scaler_min = None
_scaler_max = None

def preprocess(df, spec_cols, fit=False):
    global _scaler_min, _scaler_max
    X = np.log1p(df[spec_cols].values.astype(np.float32))
    
    if fit:
        _scaler_min = X.min(axis=0)
        _scaler_max = X.max(axis=0)
    
    # 채널별 min-max → [0, 1]
    X = (X - _scaler_min) / (_scaler_max - _scaler_min + 1e-8)
    return X


# ──────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────
class SpectrumDataset(Dataset):
    def __init__(self, X: np.ndarray):
        # shape: (N, 1, 254) — Conv1d 입력 형식
        self.X = torch.from_numpy(X).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# ──────────────────────────────────────────────
# 3. 모델: 1D-CNN Autoencoder
# ──────────────────────────────────────────────
class SpectralAutoencoder(nn.Module):
    """
    Encoder: Conv1d × 3 → Flatten → Linear → latent z
    Decoder: Linear → Unflatten → ConvTranspose1d × 3 → Sigmoid
    입력 채널 수: 254 (A0~A126 + B0~B126, Trash 제외)
    """
    def __init__(self, input_len: int = 254, latent_dim: int = 32):
        super().__init__()
        self.input_len  = input_len
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────
        self.encoder_conv = nn.Sequential(
            # (B, 1, 254) → (B, 32, 254)
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            # → (B, 64, 127)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # → (B, 128, 64)
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        # stride=2 두 번 → 254 → 127 → 64 (ceil)
        self._enc_out_len = self._calc_enc_len(input_len)
        self.encoder_fc = nn.Linear(128 * self._enc_out_len, latent_dim)

        # ── Decoder ──────────────────────────────
        self.decoder_fc = nn.Linear(latent_dim, 128 * self._enc_out_len)
        self.decoder_conv = nn.Sequential(
            # (B, 128, 64) → (B, 64, 127)
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # → (B, 32, 254)
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            # → (B, 1, 254)
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def _calc_enc_len(self, length: int) -> int:
        import math
        l = math.ceil(length / 2)   # stride=2 첫 번째
        l = math.ceil(l / 2)        # stride=2 두 번째
        return l

    def encode(self, x):
        h = self.encoder_conv(x)               # (B, 128, L)
        h = h.flatten(1)                       # (B, 128*L)
        z = self.encoder_fc(h)                 # (B, latent_dim)
        return z

    def decode(self, z):
        h = self.decoder_fc(z)                 # (B, 128*L)
        h = h.view(-1, 128, self._enc_out_len) # (B, 128, L)
        x_hat = self.decoder_conv(h)           # (B, 1, 254)
        # 길이 불일치 보정 (stride로 인한 off-by-one)
        if x_hat.shape[-1] != self.input_len:
            x_hat = x_hat[..., :self.input_len]
        return x_hat

    def forward(self, x):
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ──────────────────────────────────────────────
# 4. 손실 함수
# ──────────────────────────────────────────────
def spectral_loss(x_hat, x, alpha=0.1):
    """
    MSE + alpha × Spectral Shape Loss
    Spectral Shape Loss: 정규화된 스펙트럼의 KL divergence
    → 피크 위치와 상대적 형태 보존
    """
    mse = nn.functional.mse_loss(x_hat, x)

    # 스펙트럼 형태 보존 (softmax 후 KL divergence)
    p = torch.softmax(x.squeeze(1), dim=-1) + 1e-8
    q = torch.softmax(x_hat.squeeze(1), dim=-1) + 1e-8
    kl  = (p * (p.log() - q.log())).sum(dim=-1).mean()

    return mse + alpha * kl, mse.item(), kl.item()


# ──────────────────────────────────────────────
# 5. 학습 루프
# ──────────────────────────────────────────────
def train(model, loader, optimizer, scheduler, epochs):
    model.train()
    history = {"loss": [], "mse": [], "kl": []}

    for epoch in range(1, epochs + 1):
        ep_loss, ep_mse, ep_kl = 0., 0., 0.
        for batch in loader:
            x = batch.to(cfg.DEVICE)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss, mse, kl = spectral_loss(x_hat, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            ep_mse  += mse
            ep_kl   += kl

        n = len(loader)
        history["loss"].append(ep_loss / n)
        history["mse"].append(ep_mse / n)
        history["kl"].append(ep_kl / n)
        scheduler.step(ep_loss / n)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={ep_loss/n:.5f}  mse={ep_mse/n:.5f}  kl={ep_kl/n:.5f}")

    return history


# ──────────────────────────────────────────────
# 6. 평가 — Reconstruction Error per timestamp
# ──────────────────────────────────────────────
@torch.no_grad()
def compute_reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """샘플별 MSE 반환 (anomaly score로 사용)"""
    model.eval()
    dataset = SpectrumDataset(X)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False)
    errors  = []
    for batch in loader:
        x     = batch.to(cfg.DEVICE)
        x_hat, _ = model(x)
        mse   = ((x_hat - x) ** 2).mean(dim=(1, 2))  # (B,)
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


# ──────────────────────────────────────────────
# 7. 시각화
# ──────────────────────────────────────────────
def plot_training_history(history, save_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="Total Loss")
    axes[0].plot(history["mse"],  label="MSE",  linestyle="--")
    axes[0].plot(history["kl"],   label="KL",   linestyle=":")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].semilogy(history["loss"])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss (log)")
    axes[1].set_title("Training Loss (log scale)"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장: {save_path}")


def plot_reconstruction(model, X, spec_cols, n_samples=3,
                        save_path="reconstruction_examples.png"):
    """원본 vs 복원 스펙트럼 비교"""
    model.eval()
    indices = np.random.choice(len(X), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    # 에너지 채널 레이블 (A/B 구분)
    ch_labels = [c.split("(")[0] for c in spec_cols]  # "A5(119)" → "A5"
    x_ticks   = np.arange(len(spec_cols))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            x_np  = X[idx]
            x_t   = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(cfg.DEVICE)
            x_hat = model(x_t)[0].squeeze().cpu().numpy()

            # log1p 역변환
            orig = np.expm1(x_np)
            recon = np.expm1(x_hat)

            axes[i].bar(x_ticks, orig,  alpha=0.6, label="Original", color="steelblue", width=0.8)
            axes[i].bar(x_ticks, recon, alpha=0.6, label="Reconstructed", color="tomato", width=0.8)

            # A/B 경계선
            a_count = sum(1 for c in spec_cols if c.startswith("A"))
            axes[i].axvline(a_count - 0.5, color="black", linestyle="--", alpha=0.5, label="A|B boundary")

            mse_val = np.mean((orig - recon) ** 2)
            axes[i].set_title(f"Sample #{idx}  |  MSE={mse_val:.2f}")
            axes[i].set_xticks(x_ticks[::20])
            axes[i].set_xticklabels(ch_labels[::20], rotation=45, fontsize=7)
            axes[i].legend(fontsize=8); axes[i].grid(alpha=0.2)

    plt.suptitle("Original vs Reconstructed Spectra", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


def plot_anomaly_timeline(all_df, errors_dict, threshold_sigma=3,
                          save_path="anomaly_timeline.png"):
    """
    날짜 × PD별 reconstruction error 시계열 + 이상 탐지 threshold
    errors_dict: { (date, pd_id): np.ndarray }
    """
    dates  = sorted(cfg.ALL_DATES)
    pd_ids = [1, 2, 3]
    fig, axes = plt.subplots(len(pd_ids), 1, figsize=(15, 10), sharex=True)

    for row, pd_id in enumerate(pd_ids):
        ax = axes[row]
        all_errors = []

        for date in dates:
            key = (date, pd_id)
            if key not in errors_dict:
                continue
            errs = errors_dict[key]

            # 해당 날짜 × PD의 타임스탬프 복원
            mask = (all_df["date"] == date) & (all_df["pd_id"] == pd_id)
            times = all_df.loc[mask, "Time"].values
            if len(times) != len(errs):
                times = times[:len(errs)]

            ax.plot(times, errs, linewidth=0.7,
                    label=date, alpha=0.85)
            all_errors.extend(errs.tolist())

        # Threshold: quiet 기간(5/10~11) 통계 기반
        quiet_errors = []
        for date in cfg.TRAIN_DATES:
            key = (date, pd_id)
            if key in errors_dict:
                quiet_errors.extend(errors_dict[key].tolist())

        if quiet_errors:
            mu    = np.mean(quiet_errors)
            sigma = np.std(quiet_errors)
            thr   = mu + threshold_sigma * sigma
            ax.axhline(thr, color="red", linestyle="--", linewidth=1.2,
                       label=f"Threshold (μ+{threshold_sigma}σ={thr:.4f})")

        ax.set_ylabel(f"PD{pd_id}\nRecon. MSE", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)

        # 이벤트 기간 배경 음영
        for date in ["20240512", "20240513"]:
            key = (date, pd_id)
            if key in errors_dict:
                mask  = (all_df["date"] == date) & (all_df["pd_id"] == pd_id)
                times = all_df.loc[mask, "Time"].values
                if len(times) > 0:
                    ax.axvspan(times[0], times[-1], alpha=0.07, color="orange")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, fontsize=7)
    axes[0].set_title("KSEM PD Reconstruction Error Timeline\n"
                      "(주황 음영: 이벤트 기간 5/12~13, 적선: 이상 탐지 임계값)",
                      fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


def plot_latent_space(model, X, labels, save_path="latent_space.png"):
    """t-SNE로 latent space 시각화 (날짜별 색상)"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn 없음, latent space 시각화 스킵")
        return

    model.eval()
    dataset = SpectrumDataset(X)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)
    zs = []
    with torch.no_grad():
        for batch in loader:
            z = model.encode(batch.to(cfg.DEVICE))
            zs.append(z.cpu().numpy())
    Z = np.concatenate(zs, axis=0)

    # t-SNE: 데이터 많으면 샘플링
    max_pts = 5000
    if len(Z) > max_pts:
        idx = np.random.choice(len(Z), max_pts, replace=False)
        Z_s, lab_s = Z[idx], [labels[i] for i in idx]
    else:
        Z_s, lab_s = Z, labels

    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    Z2   = tsne.fit_transform(Z_s)

    unique_labs = sorted(set(lab_s))
    cmap = plt.cm.get_cmap("tab10", len(unique_labs))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lab in enumerate(unique_labs):
        mask = [l == lab for l in lab_s]
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=3, alpha=0.5,
                   color=cmap(i), label=lab)
    ax.legend(markerscale=5, fontsize=8)
    ax.set_title(f"Latent Space (t-SNE, dim={cfg.LATENT_DIM})")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 8. 압축률 및 성능 리포트
# ──────────────────────────────────────────────
def compression_report(model, X_all, errors_all):
    n_input  = 254                     # 채널 수
    n_latent = cfg.LATENT_DIM
    cr       = n_input / n_latent

    quiet_mask = np.array(["20240510" in l or "20240511" in l
                            for l in errors_all["label"]])
    event_mask = ~quiet_mask

    quiet_err = errors_all["error"][quiet_mask]
    event_err = errors_all["error"][event_mask]

    print("\n" + "="*50)
    print("  압축 성능 리포트")
    print("="*50)
    print(f"  입력 차원     : {n_input}")
    print(f"  잠재 차원     : {n_latent}")
    print(f"  압축률        : {cr:.1f}x")
    print(f"  모델 파라미터 : {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n  Quiet 기간 Recon. MSE : {quiet_err.mean():.5f} ± {quiet_err.std():.5f}")
    if event_mask.sum() > 0:
        print(f"  Event 기간 Recon. MSE : {event_err.mean():.5f} ± {event_err.std():.5f}")
        ratio = event_err.mean() / quiet_err.mean()
        print(f"  Event/Quiet MSE 비율  : {ratio:.2f}x  ({'이상 탐지 가능 ✓' if ratio > 2 else '차이 미미'})")
    print("="*50)

# diagnosis
def diagnose(X, spec_cols):
    print("\n[진단] 채널별 통계")
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    top5  = np.argsort(means)[-5:][::-1]
    print("  log1p 후 평균값 TOP5 채널:")
    for i in top5:
        print(f"    {spec_cols[i]:20s}  mean={means[i]:.3f}  std={stds[i]:.3f}")
    print(f"  전체 mean={means.mean():.3f}, std={stds.mean():.3f}")
    print(f"  max값: {X.max():.3f}")
# 


# ──────────────────────────────────────────────
# 9. 메인
# ──────────────────────────────────────────────
def main():
    print("\n[1] 데이터 로드")
    print("  학습용 (Quiet):")
    train_df = load_pd_data(cfg.DATA_DIR, cfg.TRAIN_DATES)
    print("  전체 (Quiet + Event):")
    all_df   = load_pd_data(cfg.DATA_DIR, cfg.ALL_DATES)

    spec_cols = get_spectral_columns(train_df)
    print(f"\n  사용 채널 수: {len(spec_cols)}")
    print(f"  예시: {spec_cols[:3]} ... {spec_cols[-3:]}")

    print("\n[2] 전처리 (log1p 정규화)")
    X_train = preprocess(train_df, spec_cols, fit=True)   # scaler fit
    X_all   = preprocess(all_df,   spec_cols, fit=False)  # 같은 scaler 적용
    diagnose(X_train, spec_cols)
    print(f"  학습 데이터 shape: {X_train.shape}")
    print(f"  전체 데이터 shape: {X_all.shape}")

    print("\n[3] 모델 초기화")
    model = SpectralAutoencoder(input_len=len(spec_cols),
                                latent_dim=cfg.LATENT_DIM).to(cfg.DEVICE)
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  인코더 출력 길이: {model._enc_out_len}  →  FC: {128 * model._enc_out_len} → {cfg.LATENT_DIM}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)
    loader    = DataLoader(SpectrumDataset(X_train),
                           batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"\n[4] 학습 ({cfg.EPOCHS} epochs, Quiet 기간만)")
    history = train(model, loader, optimizer, scheduler, cfg.EPOCHS)
    plot_training_history(history, "training_history.png")

    print("\n[5] 복원 예시 시각화")
    plot_reconstruction(model, X_all, spec_cols, n_samples=3,
                        save_path="reconstruction_examples.png")

    print("\n[6] 전체 데이터 Reconstruction Error 계산")
    errors_dict = {}
    error_list, label_list = [], []

    for date in cfg.ALL_DATES:
        for pd_id in [1, 2, 3]:
            mask = (all_df["date"] == date) & (all_df["pd_id"] == pd_id)
            if mask.sum() == 0:
                continue
            X_sub = preprocess(all_df[mask], spec_cols)
            errs  = compute_reconstruction_error(model, X_sub)
            errors_dict[(date, pd_id)] = errs
            error_list.extend(errs.tolist())
            label_list.extend([f"{date}_PD{pd_id}"] * len(errs))
            print(f"  {date} PD{pd_id}: mean MSE = {errs.mean():.5f}")

    errors_all = {
        "error": np.array(error_list),
        "label": label_list
    }

    print("\n[7] Anomaly Timeline 시각화")
    plot_anomaly_timeline(all_df, errors_dict, threshold_sigma=3,
                          save_path="anomaly_timeline.png")

    print("\n[8] Latent Space 시각화 (t-SNE)")
    tsne_labels = []
    for date in cfg.ALL_DATES:
        for pd_id in [1, 2, 3]:
            mask = (all_df["date"] == date) & (all_df["pd_id"] == pd_id)
            tsne_labels.extend([date] * mask.sum())
    plot_latent_space(model, X_all, tsne_labels, save_path="latent_space.png")

    print("\n[9] 압축 성능 리포트")
    compression_report(model, X_all, errors_all)

    print("\n[10] 모델 저장")
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "input_len":  len(spec_cols),
            "latent_dim": cfg.LATENT_DIM,
            "spec_cols":  spec_cols,
        }
    }, "ksem_autoencoder.pt")
    print("  저장: ksem_autoencoder.pt")
    print("\n완료!")


if __name__ == "__main__":
    main()