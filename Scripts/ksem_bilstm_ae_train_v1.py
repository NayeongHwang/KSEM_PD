"""
KSEM PD1 Bi-LSTM Autoencoder 압축 모델 학습
=============================================
유효 채널: A0~A41, A45~A56, B0~B41, B45~B56 (총 108채널, Proton only)

사용법:
    # window 크기 탐색 (5, 10, 30분)
    python ksem_bilstm_ae_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --search_window --max_files 100

    # k 탐색 (latent_dim=16,32,64)
    python ksem_bilstm_ae_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --search --window 10 --max_files 100

    # 최종 학습
    python ksem_bilstm_ae_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --k 32 --window 10 --save bilstm_pd1.pt

필요 패키지:
    pip install numpy pandas tqdm torch
"""

import argparse, os, glob, bz2, time
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
            bidirectional = True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
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
        self.fc_in   = nn.Linear(latent_dim, hidden_dim)
        self.lstm    = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True
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
# Window 생성
# ──────────────────────────────────────────────

def make_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """(N, 108) → (N//window_size, window_size, 108)"""
    n_windows = len(data) // window_size
    return data[:n_windows * window_size].reshape(n_windows, window_size, INPUT_DIM)


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


def load_files(files, max_files=None):
    if max_files:
        files = files[:max_files]
    arrays, failed, nan_cnt = [], 0, 0
    iterator = tqdm(files, desc="파일 로드") if USE_TQDM else files

    for f in iterator:
        try:
            df       = pd.read_csv(f)
            all_data = df.drop("Time", axis=1).values.astype(np.float32)
            if all_data.shape[1] != 256:
                continue
            arr = all_data[:, VALID_IDX]
            if np.isnan(arr).sum() > 0:
                nan_cnt += 1
                arr = arr[~np.isnan(arr).any(axis=1)]
                arr = np.nan_to_num(arr, nan=0.0)
                if len(arr) == 0:
                    continue
            arrays.append(arr)
        except Exception:
            failed += 1

    data = np.vstack(arrays)
    print(f"\n  로드 완료  : {len(arrays)}개 파일, {failed}개 실패, NaN파일 {nan_cnt}개")
    print(f"  합산 shape : {data.shape}  ({data.shape[0]//1440:.1f}일치)")
    print(f"  값 범위    : {data.min():.2f} ~ {data.max():.2f}")
    print(f"  전체 평균  : {data.mean():.2f}")
    return data


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train(train_tensor, val_tensor, latent_dim, seq_len,
          epochs=50, batch_size=256, lr=1e-3,
          device="cpu", patience=5):

    model     = BiLSTMAutoencoder(INPUT_DIM, hidden_dim=64,
                                  latent_dim=latent_dim,
                                  seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(train_tensor),
                              batch_size=batch_size, shuffle=True)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        t_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = torch.mean((xb - x_hat)**2)  # log domain MSE
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_tensor)

        model.eval()
        with torch.no_grad():
            v_hat, _ = model(val_tensor.to(device))
            v_loss   = torch.mean((val_tensor.to(device) - v_hat)**2).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:>3}/{epochs}  train={t_loss:.6f}  val={v_loss:.6f}")

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────

def evaluate(model, test_raw_win, device):
    """
    반환: CR, raw_RMSE, NRMSE, log_RMSE, weighted_RMSE
    weighted_RMSE: log1p(원본) 기반 가중치 → 고카운트 구간에 더 높은 페널티
    """
    model.eval()
    test_log = torch.tensor(preprocess(test_raw_win), dtype=torch.float32)

    with torch.no_grad():
        latent    = model.encode(test_log.to(device)).cpu().numpy()
        recon_log = model.decode(
            torch.tensor(latent, dtype=torch.float32).to(device)
        ).cpu().numpy()

    recon_raw = np.clip(postprocess(recon_log), 0, None)

    # 압축 크기
    compressed = bz2.compress(latent.astype(np.float32).tobytes(), 9)
    cr         = test_raw_win.nbytes / len(compressed)

    # 지표 계산
    diff     = test_raw_win.astype(np.float64) - recon_raw.astype(np.float64)
    raw_rmse = np.sqrt(np.mean(diff**2))
    nrmse    = raw_rmse / (test_raw_win.mean() + 1e-12)

    # log domain RMSE
    log_diff = preprocess(test_raw_win) - recon_log
    log_rmse = np.sqrt(np.mean(log_diff**2))

    # weighted RMSE
    # 가중치: log1p(원본) / mean(log1p(원본)) → 고카운트일수록 가중치 높음
    log_orig = preprocess(test_raw_win).astype(np.float64)
    weights  = log_orig / (log_orig.mean() + 1e-12)
    w_rmse   = np.sqrt(np.mean(weights * diff**2))

    return {
        "CR"           : cr,
        "raw_RMSE"     : raw_rmse,
        "NRMSE"        : nrmse,
        "log_RMSE"     : log_rmse,
        "weighted_RMSE": w_rmse,
    }


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
    parser = argparse.ArgumentParser(description="KSEM PD Bi-LSTM AE 압축 모델")
    parser.add_argument("--data_dir",      required=True)
    parser.add_argument("--pd",            default="PD1")
    parser.add_argument("--k",             type=int,   default=32)
    parser.add_argument("--window",        type=int,   default=10,  help="window 크기 (분)")
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--num_layers",    type=int,   default=1)
    parser.add_argument("--search",        action="store_true", help="k 탐색 (16,32,64)")
    parser.add_argument("--search_window", action="store_true", help="window 탐색 (5,10,30분)")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--test_ratio",    type=float, default=0.1)
    parser.add_argument("--max_files",     type=int,   default=None)
    parser.add_argument("--save",          default="bilstm_model.pt")
    parser.add_argument("--device",        default="auto")
    args = parser.parse_args()

    # 디바이스
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'='*65}")
    print(f"  KSEM {args.pd} Bi-LSTM Autoencoder 압축 모델")
    print(f"  유효 채널  : {INPUT_DIM}채널")
    print(f"  전처리     : log1p 변환")
    print(f"  디바이스   : {device}")
    print(f"{'='*65}")

    # 파일 탐색
    print(f"\n[1] 파일 탐색")
    files = find_files(args.data_dir, args.pd)
    if not files:
        print(f"  [오류] 파일 없음: {args.data_dir}")
        return
    print(f"  발견: {len(files)}개 파일")
    print(f"  기간: {os.path.basename(files[0])[:8]} ~ {os.path.basename(files[-1])[:8]}")

    # 데이터 로드
    print(f"\n[2] 데이터 로드 (max_files={args.max_files})")
    data = load_files(files, max_files=args.max_files)

    # ── window 탐색 모드 ──
    if args.search_window:
        search_windows = [5, 10, 30]
        k_fixed        = args.k

        print(f"\n[3] Window 탐색: {search_windows}분  (k={k_fixed} 고정)")
        print(f"\n{'='*80}")
        print(f"  {'win':>5} {'n_win':>7} {'CR':>8} {'raw_RMSE':>10} "
              f"{'NRMSE':>10} {'log_RMSE':>10} {'w_RMSE':>10}")
        print(f"{'='*80}")

        for w in search_windows:
            windows = make_windows(data, w)
            n       = len(windows)
            n_test  = max(1, int(n * args.test_ratio))
            n_val   = max(1, int(n * 0.1))
            n_train = n - n_test - n_val

            train_t = torch.tensor(preprocess(windows[:n_train]),           dtype=torch.float32)
            val_t   = torch.tensor(preprocess(windows[n_train:n_train+n_val]), dtype=torch.float32)
            test_raw= windows[n_train+n_val:]

            print(f"\n  --- window={w}분  (train {n_train}개 / val {n_val}개 / test {n_test}개) ---")
            model = train(train_t, val_t, k_fixed, w,
                          epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, device=device, patience=args.patience)

            m = evaluate(model, test_raw, device)
            print(f"  {w:>5} {n:>7} {m['CR']:>8.2f}x {m['raw_RMSE']:>10.4f} "
                  f"{m['NRMSE']:>10.6f} {m['log_RMSE']:>10.6f} {m['weighted_RMSE']:>10.4f}")

        print(f"\n{'='*80}")
        print(f"  탐색 완료. 최적 window 선택 후 --search 또는 단일 학습 진행")
        return

    # ── k 탐색 모드 ──
    windows = make_windows(data, args.window)
    print(f"\n[3] Window 생성 (window={args.window}분)")
    print(f"  windows shape: {windows.shape}")

    n       = len(windows)
    n_test  = max(1, int(n * args.test_ratio))
    n_val   = max(1, int(n * 0.1))
    n_train = n - n_test - n_val

    train_t  = torch.tensor(preprocess(windows[:n_train]),              dtype=torch.float32)
    val_t    = torch.tensor(preprocess(windows[n_train:n_train+n_val]), dtype=torch.float32)
    test_raw = windows[n_train+n_val:]

    print(f"\n[4] Train/Val/Test 분리")
    print(f"  Train: {train_t.shape}  Val: {val_t.shape}  Test: {test_raw.shape}")

    search_ks = [16, 32, 64] if args.search else [args.k]

    if args.search:
        print(f"\n[5] k 탐색: {search_ks}  (window={args.window}분)")
        print(f"\n{'='*80}")
        print(f"  {'k':>4} {'CR':>8} {'raw_RMSE':>10} {'NRMSE':>10} "
              f"{'log_RMSE':>10} {'w_RMSE':>10}")
        print(f"{'='*80}")

    for k in search_ks:
        print(f"\n  --- k={k} 학습 중 ---")
        t0    = time.perf_counter()
        model = train(train_t, val_t, k, args.window,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, device=device, patience=args.patience)
        print(f"  학습 시간: {time.perf_counter()-t0:.1f}초")

        m = evaluate(model, test_raw, device)

        if args.search:
            print(f"  {k:>4} {m['CR']:>8.2f}x {m['raw_RMSE']:>10.4f} "
                  f"{m['NRMSE']:>10.6f} {m['log_RMSE']:>10.6f} {m['weighted_RMSE']:>10.4f}")
        else:
            print(f"\n[6] 테스트 평가")
            print(f"  CR            : {m['CR']:.2f}x")
            print(f"  raw RMSE      : {m['raw_RMSE']:.4f}")
            print(f"  NRMSE         : {m['NRMSE']:.6f}  ({m['NRMSE']*100:.2f}%)")
            print(f"  log RMSE      : {m['log_RMSE']:.6f}")
            print(f"  weighted RMSE : {m['weighted_RMSE']:.4f}")
            print(f"\n[7] 모델 저장")
            save_model(args.save, model, k, args.window)

    if args.search:
        print(f"\n{'='*80}")
        print(f"  탐색 완료. 최적 k 선택 후:")
        print(f"  python ksem_bilstm_ae_train.py --data_dir ... --pd {args.pd} --window {args.window} --k <k> --save bilstm_pd1.pt")

    print(f"\n완료!\n")


if __name__ == "__main__":
    main()