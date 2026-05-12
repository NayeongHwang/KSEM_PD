"""
KSEM PD1 Bi-LSTM Autoencoder 압축 모델 학습 (개선판)
=====================================================
개선 사항:
    1. ReduceLROnPlateau 학습률 스케줄러
    2. log MSE + weighted MSE 조합 loss
    3. hidden_dim, num_layers 조정 가능
    4. 파일(날짜) 단위 train/val/test 분리

사용법:
    # window 탐색
    python ksem_bilstm_ae_train.py --data_dir D:\\workspace\\KSEM_L0_PD1\\Raw_count\\Raw_count --pd PD1 --search_window --k 16

    # k 탐색
    python ksem_bilstm_ae_train.py --data_dir D:\\workspace\\KSEM_L0_PD1\\Raw_count\\Raw_count --pd PD1 --search --window 10

    # 최종 학습
    python ksem_bilstm_ae_train.py --data_dir D:\\workspace\\KSEM_L0_PD1\\Raw_count\\Raw_count --pd PD1 --k 16 --window 10 --epochs 200 --patience 20 --save bilstm_pd1.pt

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
# 모델 정의
# ──────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=108, hidden_dim=64, latent_dim=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = 0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim=108, hidden_dim=64, latent_dim=16,
                 seq_len=10, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in   = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm    = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1 if num_layers > 1 else 0.0
        )
        self.fc_out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h        = self.fc_in(z)
        repeated = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec, _   = self.lstm(repeated)
        return self.fc_out(dec)


class BiLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=108, hidden_dim=64, latent_dim=16,
                 seq_len=10, num_layers=1):
        super().__init__()
        self.encoder = BiLSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, latent_dim, seq_len, num_layers)

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
# Loss 함수
# ──────────────────────────────────────────────

def weighted_loss(x_log, x_hat_log, w_coeff=0.1):
    """
    log MSE + weighted MSE 조합
    - log MSE: log 도메인에서 전반적인 오차
    - weighted MSE: 고카운트(폭풍 날) 구간에 더 높은 페널티
    """
    log_mse = torch.mean((x_log - x_hat_log) ** 2)

    # 가중치: log 값이 클수록 (고카운트일수록) 가중치 높음
    weights  = x_log / (x_log.mean() + 1e-8)
    w_mse    = torch.mean(weights * (x_log - x_hat_log) ** 2)

    return log_mse + w_coeff * w_mse


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


def make_windows_from_files(files, window_size, desc="로드"):
    all_windows = []
    iterator    = tqdm(files, desc=desc) if USE_TQDM else files
    for f in iterator:
        try:
            arr = load_file(f)
            if arr is None:
                continue
            n_win = len(arr) // window_size
            if n_win == 0:
                continue
            windows = arr[:n_win * window_size].reshape(n_win, window_size, INPUT_DIM)
            all_windows.append(windows)
        except Exception:
            pass
    if not all_windows:
        return np.zeros((0, window_size, INPUT_DIM), dtype=np.float32)
    return np.concatenate(all_windows, axis=0)


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train_model(train_tensor, val_tensor, latent_dim, seq_len,
                hidden_dim=64, num_layers=1,
                epochs=100, batch_size=256, lr=1e-3,
                device="cpu", patience=10, w_coeff=0.1):

    model     = BiLSTMAutoencoder(INPUT_DIM, hidden_dim=hidden_dim,
                                  latent_dim=latent_dim,
                                  seq_len=seq_len,
                                  num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ReduceLROnPlateau: val_loss 개선 없으면 lr 줄임
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    loader = DataLoader(TensorDataset(train_tensor),
                        batch_size=batch_size, shuffle=True)

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
            loss      = weighted_loss(xb, x_hat, w_coeff)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_tensor)

        model.eval()
        with torch.no_grad():
            v_hat, _ = model(val_tensor.to(device))
            v_loss   = weighted_loss(val_tensor.to(device), v_hat, w_coeff).item()

        # lr 스케줄러 업데이트
        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:>4}/{epochs}  train={t_loss:.6f}  val={v_loss:.6f}  lr={current_lr:.2e}")

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

def evaluate(model, test_raw_win, device):
    model.eval()
    test_log = torch.tensor(preprocess(test_raw_win), dtype=torch.float32)

    with torch.no_grad():
        latent    = model.encode(test_log.to(device)).cpu().numpy()
        recon_log = model.decode(
            torch.tensor(latent, dtype=torch.float32).to(device)
        ).cpu().numpy()

    recon_raw  = np.clip(postprocess(recon_log), 0, None)
    compressed = bz2.compress(latent.astype(np.float32).tobytes(), 9)
    cr         = test_raw_win.nbytes / len(compressed)

    diff     = test_raw_win.astype(np.float64) - recon_raw.astype(np.float64)
    raw_rmse = np.sqrt(np.mean(diff**2))
    nrmse    = raw_rmse / (test_raw_win.mean() + 1e-12)

    log_diff = preprocess(test_raw_win) - recon_log
    log_rmse = np.sqrt(np.mean(log_diff**2))

    log_orig = preprocess(test_raw_win).astype(np.float64)
    weights  = log_orig / (log_orig.mean() + 1e-12)
    w_rmse   = np.sqrt(np.mean(weights * diff**2))

    return {"CR": cr, "raw_RMSE": raw_rmse, "NRMSE": nrmse,
            "log_RMSE": log_rmse, "weighted_RMSE": w_rmse}


# ──────────────────────────────────────────────
# 모델 저장
# ──────────────────────────────────────────────

def save_model(path, model, latent_dim, seq_len):
    torch.save({
        "latent_dim": latent_dim,
        "seq_len"   : seq_len,
        "input_dim" : INPUT_DIM,
        "state_dict": model.state_dict(),
        "valid_idx" : VALID_IDX,
    }, path)
    print(f"  모델 저장  : {path}  ({os.path.getsize(path)/1024:.1f} KB)")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KSEM PD Bi-LSTM AE (개선판)")
    parser.add_argument("--data_dir",      required=True)
    parser.add_argument("--pd",            default="PD1")
    parser.add_argument("--k",             type=int,   default=16)
    parser.add_argument("--window",        type=int,   default=10)
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--num_layers",    type=int,   default=1)
    parser.add_argument("--search",        action="store_true", help="k 탐색 (16,32,64)")
    parser.add_argument("--search_window", action="store_true", help="window 탐색 (5,10,30분)")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--w_coeff",       type=float, default=0.1,
                        help="weighted loss 계수 (기본: 0.1)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--max_files",     type=int,   default=None)
    parser.add_argument("--save",          default="bilstm_model.pt")
    parser.add_argument("--device",        default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'='*65}")
    print(f"  KSEM {args.pd} Bi-LSTM AE (개선판)")
    print(f"  유효 채널  : {INPUT_DIM}채널")
    print(f"  전처리     : log1p 변환")
    print(f"  loss       : log MSE + {args.w_coeff} × weighted MSE")
    print(f"  스케줄러   : ReduceLROnPlateau (factor=0.5, patience=5)")
    print(f"  디바이스   : {device}")
    print(f"{'='*65}")

    print(f"\n[1] 파일 탐색")
    files = find_files(args.data_dir, args.pd)
    if not files:
        print(f"  [오류] 파일 없음: {args.data_dir}")
        return
    if args.max_files:
        files = files[:args.max_files]
    print(f"  발견: {len(files)}개 파일")
    print(f"  기간: {os.path.basename(files[0])[:8]} ~ {os.path.basename(files[-1])[:8]}")

    print(f"\n[2] 파일 단위 Train/Val/Test 분리 (seed={args.seed})")
    train_files, val_files, test_files = split_files(files, seed=args.seed)
    print(f"  Train: {len(train_files)}개 / Val: {len(val_files)}개 / Test: {len(test_files)}개")

    # ── window 탐색 ──
    if args.search_window:
        search_windows = [5, 10, 30]
        print(f"\n[3] Window 탐색: {search_windows}분  (k={args.k} 고정)")
        print(f"\n{'='*82}")
        print(f"  {'win':>5} {'CR':>8} {'raw_RMSE':>10} {'NRMSE':>10} {'log_RMSE':>10} {'w_RMSE':>10}")
        print(f"{'='*82}")

        for w in search_windows:
            print(f"\n  --- window={w}분 ---")
            train_w = make_windows_from_files(train_files, w, f"  train w={w}")
            val_w   = make_windows_from_files(val_files,   w, f"  val   w={w}")
            test_w  = make_windows_from_files(test_files,  w, f"  test  w={w}")
            train_t = torch.tensor(preprocess(train_w), dtype=torch.float32)
            val_t   = torch.tensor(preprocess(val_w),   dtype=torch.float32)

            model = train_model(train_t, val_t, args.k, w,
                                hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, device=device, patience=args.patience,
                                w_coeff=args.w_coeff)
            m = evaluate(model, test_w, device)
            print(f"  {w:>5} {m['CR']:>8.2f}x {m['raw_RMSE']:>10.4f} "
                  f"{m['NRMSE']:>10.6f} {m['log_RMSE']:>10.6f} {m['weighted_RMSE']:>10.4f}")

        print(f"\n{'='*82}")
        return

    # ── window 생성 ──
    w = args.window
    print(f"\n[3] Window 생성 (window={w}분)")
    train_w = make_windows_from_files(train_files, w, "  train 로드")
    val_w   = make_windows_from_files(val_files,   w, "  val   로드")
    test_w  = make_windows_from_files(test_files,  w, "  test  로드")
    print(f"  train:{train_w.shape} val:{val_w.shape} test:{test_w.shape}")

    train_t = torch.tensor(preprocess(train_w), dtype=torch.float32)
    val_t   = torch.tensor(preprocess(val_w),   dtype=torch.float32)

    search_ks = [16, 32, 64] if args.search else [args.k]

    if args.search:
        print(f"\n[4] k 탐색: {search_ks}  (window={w}분)")
        print(f"\n{'='*80}")
        print(f"  {'k':>4} {'CR':>8} {'raw_RMSE':>10} {'NRMSE':>10} {'log_RMSE':>10} {'w_RMSE':>10}")
        print(f"{'='*80}")

    for k in search_ks:
        print(f"\n  --- k={k} 학습 중 ---")
        t0    = time.perf_counter()
        model = train_model(train_t, val_t, k, w,
                            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                            epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, device=device, patience=args.patience,
                            w_coeff=args.w_coeff)
        print(f"  학습 시간: {time.perf_counter()-t0:.1f}초")
        m = evaluate(model, test_w, device)

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
            save_model(args.save, model, k, w)

    if args.search:
        print(f"\n{'='*80}")
        print(f"  탐색 완료.")

    print(f"\n완료!\n")


if __name__ == "__main__":
    main()