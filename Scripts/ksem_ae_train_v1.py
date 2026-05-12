"""
KSEM PD1 수년치 데이터 Autoencoder 압축 모델 학습
==================================================
유효 채널: A0~A41, A45~A56, B0~B41, B45~B56 (총 108채널, Proton only)

구조:
    인코더: FC(108→64→32→latent_dim)
    디코더: FC(latent_dim→32→64→108)
    활성함수: ReLU (출력층 제외)
    전처리: log1p 변환 (권장)

사용법:
    # k 탐색 (여러 latent_dim 비교)
    python ksem_ae_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --search

    # 특정 k로 학습
    python ksem_ae_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --k 16 --save ae_pd1.pt

    # 빠른 테스트
    python ksem_ae_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --search --max_files 100

필요 패키지:
    pip install numpy pandas scikit-learn tqdm torch
"""

import argparse, os, glob, bz2, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
VALID_IDX   = A_VALID_IDX + B_VALID_IDX  # 총 108채널

INPUT_DIM = len(VALID_IDX)  # 108


# ──────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ──────────────────────────────────────────────
# 전처리 / 역처리
# ──────────────────────────────────────────────

def preprocess(data: np.ndarray) -> np.ndarray:
    return np.log1p(data)

def postprocess(data: np.ndarray) -> np.ndarray:
    return np.expm1(data)


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
    arrays, failed, nan_files = [], 0, []
    iterator = tqdm(files, desc="파일 로드") if USE_TQDM else files

    for f in iterator:
        try:
            df       = pd.read_csv(f)
            all_data = df.drop("Time", axis=1).values.astype(np.float32)
            if all_data.shape[1] != 256:
                continue
            arr = all_data[:, VALID_IDX]
            if np.isnan(arr).sum() > 0:
                nan_files.append(os.path.basename(f))
                arr = arr[~np.isnan(arr).any(axis=1)]
                arr = np.nan_to_num(arr, nan=0.0)
                if len(arr) == 0:
                    continue
            arrays.append(arr)
        except Exception:
            failed += 1

    data = np.vstack(arrays)
    print(f"\n  로드 완료  : {len(arrays)}개 파일, {failed}개 실패, NaN파일 {len(nan_files)}개")
    print(f"  합산 shape : {data.shape}  ({data.shape[0]//1440:.1f}일치)")
    print(f"  값 범위    : {data.min():.2f} ~ {data.max():.2f}")
    print(f"  전체 평균  : {data.mean():.2f}")
    return data


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train_ae(train_tensor, val_tensor, latent_dim,
             epochs=50, batch_size=1024, lr=1e-3,
             device="cpu", patience=5):

    model     = Autoencoder(INPUT_DIM, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(train_tensor),
                              batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs+1):
        # 학습
        model.train()
        train_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_tensor)

        # 검증
        model.eval()
        with torch.no_grad():
            val_out  = model(val_tensor.to(device))
            val_loss = criterion(val_out, val_tensor.to(device)).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────

def evaluate(model, test_raw, raw_size, device):
    """
    압축: 원본 → log1p → 정규화 → 인코더 → latent → bz2
    복원: bz2 → latent → 디코더 → 역정규화 → expm1
    평가: 원본 카운트 기준 RMSE, NRMSE, CR
    """
    model.eval()
    test_proc = torch.tensor(preprocess(test_raw), dtype=torch.float32)

    with torch.no_grad():
        latent = model.encode(test_proc.to(device)).cpu().numpy()
        recon_proc = model.decode(
            torch.tensor(latent, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # 역변환
    recon_raw = np.clip(postprocess(recon_proc), 0, None).astype(np.float32)

    # 압축 크기 (latent + bz2)
    compressed = bz2.compress(latent.astype(np.float32).tobytes(), 9)
    comp_size  = len(compressed)

    # 지표
    diff  = test_raw.astype(np.float64) - recon_raw.astype(np.float64)
    rmse  = np.sqrt(np.mean(diff**2))
    nrmse = rmse / (test_raw.mean() + 1e-12)
    cr    = raw_size / comp_size

    return {"CR": cr, "RMSE": rmse, "NRMSE": nrmse}


# ──────────────────────────────────────────────
# 모델 저장
# ──────────────────────────────────────────────

def save_model(path, model, latent_dim):
    torch.save({
        "latent_dim" : latent_dim,
        "input_dim"  : INPUT_DIM,
        "state_dict" : model.state_dict(),
        "valid_idx"  : VALID_IDX,
    }, path)
    print(f"  모델 저장  : {path}  ({os.path.getsize(path)/1024:.1f} KB)")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KSEM PD AE 압축 모델 학습")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--pd",         default="PD1")
    parser.add_argument("--k",          type=int, default=16,  help="latent 차원 (기본: 16)")
    parser.add_argument("--search",     action="store_true",   help="k 탐색 모드 (k=4,8,12,16,20)")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int, default=5,   help="Early stopping patience")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_files",  type=int, default=None)
    parser.add_argument("--save",       default="ae_model.pt")
    parser.add_argument("--device",     default="auto",        help="cpu / cuda / auto")
    args = parser.parse_args()

    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\n{'='*55}")
    print(f"  KSEM {args.pd} Autoencoder 압축 모델 학습")
    print(f"  유효 채널 : {INPUT_DIM}채널 (A:{len(A_VALID_IDX)} + B:{len(B_VALID_IDX)})")
    print(f"  전처리    : log1p 변환 적용")
    print(f"  디바이스  : {device}")
    print(f"{'='*55}")

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

    # Train/Test 분리
    print(f"\n[3] Train/Val/Test 분리")
    n_test  = max(1440, int(len(data) * args.test_ratio))
    n_val   = max(1440, int(len(data) * 0.1))
    n_train = len(data) - n_test - n_val
    train_raw = data[:n_train]
    val_raw   = data[n_train:n_train+n_val]
    test_raw  = data[n_train+n_val:]
    print(f"  Train: {train_raw.shape}  ({n_train//1440:.1f}일치)")
    print(f"  Val  : {val_raw.shape}    ({n_val//1440:.1f}일치)")
    print(f"  Test : {test_raw.shape}   ({n_test//1440:.1f}일치)")

    # 전처리 → 텐서 변환
    train_tensor = torch.tensor(preprocess(train_raw), dtype=torch.float32)
    val_tensor   = torch.tensor(preprocess(val_raw),   dtype=torch.float32)

    # k 탐색 or 단일 학습
    search_ks = [4, 8, 12, 16, 20] if args.search else [args.k]

    if args.search:
        print(f"\n[4] k 탐색: {search_ks}")
        print(f"\n{'='*60}")
        print(f"  {'k':>4} {'CR':>8} {'RMSE':>10} {'NRMSE':>10}")
        print(f"{'='*60}")

    for k in search_ks:
        if args.search:
            print(f"\n  --- k={k} 학습 중 ---")
        else:
            print(f"\n[4] AE 학습 (k={k})")

        t0    = time.perf_counter()
        model = train_ae(train_tensor, val_tensor, k,
                         epochs=args.epochs,
                         batch_size=args.batch_size,
                         lr=args.lr,
                         device=device,
                         patience=args.patience)
        t_train = time.perf_counter() - t0
        print(f"  학습 시간 : {t_train:.1f}초")

        m = evaluate(model, test_raw, test_raw.nbytes, device)

        if args.search:
            print(f"  {'k':>4} {'CR':>8} {'RMSE':>10} {'NRMSE':>10}")
            print(f"  {k:>4} {m['CR']:>8.2f}x {m['RMSE']:>10.4f} {m['NRMSE']:>10.6f}")
        else:
            print(f"\n[5] 테스트 평가")
            print(f"  압축률 (CR) : {m['CR']:.2f}x")
            print(f"  RMSE        : {m['RMSE']:.4f}")
            print(f"  NRMSE       : {m['NRMSE']:.6f}  ({m['NRMSE']*100:.2f}%)")
            print(f"\n[6] 모델 저장")
            save_model(args.save, model, k)

    if args.search:
        print(f"\n{'='*60}")
        print(f"  탐색 완료. 최적 k 선택 후 아래 명령어로 학습:")
        print(f"  python ksem_ae_train.py --data_dir ... --pd {args.pd} --k <선택한k> --save ae_pd1.pt")

    print(f"\n완료!\n")


if __name__ == "__main__":
    main()