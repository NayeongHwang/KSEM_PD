"""
KSEM PD1 1D-CNN Autoencoder 압축 모델 학습
==========================================
유효 채널: A0~A41, A45~A56, B0~B41, B45~B56 (총 108채널, Proton only)

구조:
    인코더: Conv1d(1→16→32) → Flatten → FC → latent
    디코더: FC → Reshape → ConvTranspose1d(32→16→1)
    전처리: log1p 변환
    입력:   1분 단위 (108채널을 1D 시퀀스로 처리)

사용법:
    # k 탐색 (8, 16, 32)
    python ksem_cnn_ae_train.py --data_dir D:\\workspace\\KSEM_L0_PD1\\Raw_count\\Raw_count --pd PD1 --search

    # 최종 학습
    python ksem_cnn_ae_train.py --data_dir D:\\workspace\\KSEM_L0_PD1\\Raw_count\\Raw_count --pd PD1 --k 16 --save cnn_ae_pd1.pt

필요 패키지:
    pip install numpy pandas tqdm torch
"""

import argparse, os, glob, bz2, time, random
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 유효 채널 정의 ──
A_VALID_IDX = [i for i in range(57) if i not in (42, 43, 44)]
B_VALID_IDX = [128 + i for i in range(57) if i not in (42, 43, 44)]
VALID_IDX   = A_VALID_IDX + B_VALID_IDX
INPUT_DIM   = len(VALID_IDX)  # 108


# ──────────────────────────────────────────────
# 1D-CNN AE 모델 정의
# ──────────────────────────────────────────────
# 입력: (B, 1, 108) — 108채널을 길이 108짜리 1D 시퀀스로 처리
# Conv1d가 에너지 채널 간 지역적 패턴(인접 에너지 bin 관계)을 학습

class CNNEncoder(nn.Module):
    def __init__(self, input_dim=108, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 1, 108) → (B, 16, 54)
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # (B, 16, 54) → (B, 32, 27)
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # (B, 32, 27) → (B, 64, 14)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # conv 출력 크기 계산
        self._conv_out = self._get_conv_out(input_dim)
        self.fc = nn.Linear(self._conv_out, latent_dim)

    def _get_conv_out(self, input_dim):
        dummy = torch.zeros(1, 1, input_dim)
        out   = self.conv(dummy)
        return int(out.numel())

    def forward(self, x):
        # x: (B, 108) → (B, 1, 108)
        x = x.unsqueeze(1)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=16, conv_out_size=896, output_dim=108):
        super().__init__()
        self.fc          = nn.Linear(latent_dim, conv_out_size)
        self.conv_out_ch = 64
        self.conv_out_L  = conv_out_size // self.conv_out_ch

        self.deconv = nn.Sequential(
            # (B, 64, L) → (B, 32, L*2)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # (B, 32, L*2) → (B, 16, L*4)
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # (B, 16, L*4) → (B, 1, L*8)
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )
        self.output_dim = output_dim

    def forward(self, z):
        h   = self.fc(z)
        h   = h.view(h.size(0), self.conv_out_ch, self.conv_out_L)
        out = self.deconv(h)
        out = out.squeeze(1)
        # 크기 맞춤 (패딩/크롭)
        if out.size(1) > self.output_dim:
            out = out[:, :self.output_dim]
        elif out.size(1) < self.output_dim:
            pad = torch.zeros(out.size(0), self.output_dim - out.size(1), device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out


class CNNAutoencoder(nn.Module):
    def __init__(self, input_dim=108, latent_dim=16):
        super().__init__()
        self.encoder    = CNNEncoder(input_dim, latent_dim)
        conv_out_size   = self.encoder._conv_out
        self.decoder    = CNNDecoder(latent_dim, conv_out_size, input_dim)

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)


# ──────────────────────────────────────────────
# 전처리 / 역처리
# ──────────────────────────────────────────────

def preprocess(data):
    return np.log1p(data).astype(np.float32)

def postprocess(data):
    return np.expm1(data).astype(np.float32)


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

def find_files(data_dir, pd_name):
    pattern = os.path.join(data_dir, "*", f"*_{pd_name}_Raw Count.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(data_dir, "*", f"*_{pd_name}_Raw_Count.csv")
        files   = sorted(glob.glob(pattern))
    return files


def load_file(f):
    df       = pd.read_csv(f)
    all_data = df.drop("Time", axis=1).values.astype(np.float32)
    if all_data.shape[1] != 256:
        return None
    arr = all_data[:, VALID_IDX]
    if np.isnan(arr).sum() > 0:
        arr = arr[~np.isnan(arr).any(axis=1)]
        arr = np.nan_to_num(arr, nan=0.0)
    return arr if len(arr) > 0 else None


def split_files(files, train_ratio=0.8, val_ratio=0.1, seed=42):
    files = files.copy()
    random.seed(seed)
    random.shuffle(files)
    n       = len(files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return sorted(files[:n_train]), sorted(files[n_train:n_train+n_val]), sorted(files[n_train+n_val:])


def load_files(files, desc="로드"):
    arrays = []
    iterator = tqdm(files, desc=desc) if USE_TQDM else files
    for f in iterator:
        try:
            arr = load_file(f)
            if arr is not None:
                arrays.append(arr)
        except Exception:
            pass
    return np.vstack(arrays) if arrays else np.zeros((0, INPUT_DIM), dtype=np.float32)


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train_model(train_tensor, val_tensor, latent_dim,
                epochs=100, batch_size=1024, lr=1e-3,
                device="cpu", patience=10):

    model     = CNNAutoencoder(INPUT_DIM, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    loader    = DataLoader(TensorDataset(train_tensor),
                           batch_size=batch_size, shuffle=True)

    # 모델 파라미터 수 출력
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    모델 파라미터 수: {n_params:,}")

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        t_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss      = criterion(x_hat, xb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_tensor)

        model.eval()
        with torch.no_grad():
            v_hat, _ = model(val_tensor.to(device))
            v_loss   = criterion(v_hat, val_tensor.to(device)).item()

        scheduler.step(v_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:>4}/{epochs}  train={t_loss:.6f}  val={v_loss:.6f}  lr={lr_now:.2e}")

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}  (best_val={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────

def evaluate(model, test_raw, device):
    model.eval()
    test_log = torch.tensor(preprocess(test_raw), dtype=torch.float32)

    with torch.no_grad():
        latent    = model.encode(test_log.to(device)).cpu().numpy()
        recon_log = model.decode(
            torch.tensor(latent, dtype=torch.float32).to(device)
        ).cpu().numpy()

    recon_raw  = np.clip(postprocess(recon_log), 0, None)
    compressed = bz2.compress(latent.astype(np.float32).tobytes(), 9)
    cr         = test_raw.nbytes / len(compressed)

    diff     = test_raw.astype(np.float64) - recon_raw.astype(np.float64)
    raw_rmse = np.sqrt(np.mean(diff**2))
    nrmse    = raw_rmse / (test_raw.mean() + 1e-12)

    log_diff = preprocess(test_raw) - recon_log
    log_rmse = np.sqrt(np.mean(log_diff**2))

    log_orig = preprocess(test_raw).astype(np.float64)
    weights  = log_orig / (log_orig.mean() + 1e-12)
    w_rmse   = np.sqrt(np.mean(weights * diff**2))

    return {"CR": cr, "raw_RMSE": raw_rmse, "NRMSE": nrmse,
            "log_RMSE": log_rmse, "weighted_RMSE": w_rmse}


# ──────────────────────────────────────────────
# 모델 저장
# ──────────────────────────────────────────────

def save_model(path, model, latent_dim):
    torch.save({
        "latent_dim": latent_dim,
        "input_dim" : INPUT_DIM,
        "state_dict": model.state_dict(),
        "valid_idx" : VALID_IDX,
    }, path)
    print(f"  모델 저장: {path}  ({os.path.getsize(path)/1024:.1f} KB)")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KSEM PD 1D-CNN AE 압축 모델")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--pd",         default="PD1")
    parser.add_argument("--k",          type=int,   default=16,  help="latent 차원")
    parser.add_argument("--search",     action="store_true",     help="k 탐색 (8,16,32)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=1024)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--max_files",  type=int,   default=None)
    parser.add_argument("--save",       default="cnn_ae_model.pt")
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'='*60}")
    print(f"  KSEM {args.pd} 1D-CNN Autoencoder 압축 모델")
    print(f"  유효 채널  : {INPUT_DIM}채널")
    print(f"  전처리     : log1p 변환")
    print(f"  입력 형태  : (B, 1, 108) — 에너지 채널을 1D 시퀀스로")
    print(f"  디바이스   : {device}")
    print(f"{'='*60}")

    # 파일 탐색
    print(f"\n[1] 파일 탐색")
    files = find_files(args.data_dir, args.pd)
    if not files:
        print(f"  [오류] 파일 없음: {args.data_dir}")
        return
    if args.max_files:
        files = files[:args.max_files]
    print(f"  발견: {len(files)}개 파일")
    print(f"  기간: {os.path.basename(files[0])[:8]} ~ {os.path.basename(files[-1])[:8]}")

    # 파일 단위 분리
    print(f"\n[2] Train/Val/Test 분리 (seed={args.seed})")
    train_files, val_files, test_files = split_files(files, seed=args.seed)
    print(f"  Train: {len(train_files)}개 / Val: {len(val_files)}개 / Test: {len(test_files)}개")

    # 데이터 로드
    print(f"\n[3] 데이터 로드")
    train_raw = load_files(train_files, "  train 로드")
    val_raw   = load_files(val_files,   "  val   로드")
    test_raw  = load_files(test_files,  "  test  로드")
    print(f"  train:{train_raw.shape}  val:{val_raw.shape}  test:{test_raw.shape}")
    print(f"  값 범위: {train_raw.min():.2f} ~ {train_raw.max():.2f}")

    # 텐서 변환
    train_t = torch.tensor(preprocess(train_raw), dtype=torch.float32)
    val_t   = torch.tensor(preprocess(val_raw),   dtype=torch.float32)

    # k 탐색
    search_ks = [8, 16, 32] if args.search else [args.k]

    if args.search:
        print(f"\n[4] k 탐색: {search_ks}")
        print(f"\n{'='*75}")
        print(f"  {'k':>4} {'CR':>8} {'raw_RMSE':>10} {'NRMSE':>10} {'log_RMSE':>10} {'w_RMSE':>10}")
        print(f"{'='*75}")

    for k in search_ks:
        print(f"\n  --- k={k} 학습 중 ---")
        t0    = time.perf_counter()
        model = train_model(train_t, val_t, k,
                            epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, device=device, patience=args.patience)
        print(f"  학습 시간: {time.perf_counter()-t0:.1f}초")
        m = evaluate(model, test_raw, device)

        if args.search:
            print(f"  {k:>4} {m['CR']:>8.2f}x {m['raw_RMSE']:>10.4f} "
                  f"{m['NRMSE']:>10.6f} {m['log_RMSE']:>10.6f} {m['weighted_RMSE']:>10.4f}")
        else:
            print(f"\n[5] 테스트 평가")
            print(f"  CR            : {m['CR']:.2f}x")
            print(f"  raw RMSE      : {m['raw_RMSE']:.4f}")
            print(f"  NRMSE         : {m['NRMSE']:.6f}  ({m['NRMSE']*100:.2f}%)")
            print(f"  log RMSE      : {m['log_RMSE']:.6f}")
            print(f"  weighted RMSE : {m['weighted_RMSE']:.4f}")
            print(f"\n[6] 모델 저장")
            save_model(args.save, model, k)

    if args.search:
        print(f"\n{'='*75}")
        print(f"  탐색 완료. 최적 k 선택 후:")
        print(f"  python ksem_cnn_ae_train.py --data_dir ... --pd {args.pd} --k <k> --save cnn_ae_pd1.pt")

    print(f"\n완료!\n")


if __name__ == "__main__":
    main()
