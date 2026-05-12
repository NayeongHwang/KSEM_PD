"""
KSEM PD1 수년치 데이터 PCA 압축 모델 학습
==========================================
유효 채널: A0~A41, A45~A56, B0~B41, B45~B56 (총 108채널, Proton only)

사용법:
    python ksem_pca_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --search --log
    python ksem_pca_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --search --log --k_max 50
    python ksem_pca_train.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --k 23 --log --save pca_pd1.npz

필요 패키지:
    pip install numpy pandas scikit-learn tqdm
"""

import argparse, os, glob, bz2, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ── 유효 채널 정의 ──
A_VALID_IDX = [i for i in range(57) if i not in (42, 43, 44)]
B_VALID_IDX = [128 + i for i in range(57) if i not in (42, 43, 44)]
VALID_IDX   = A_VALID_IDX + B_VALID_IDX  # 총 108채널


def preprocess(data, use_log):
    return np.log1p(data) if use_log else data.copy()

def postprocess(data, use_log):
    return np.expm1(data) if use_log else data


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

    if not arrays:
        raise ValueError("로드된 파일이 없음")

    data = np.vstack(arrays)
    print(f"\n  로드 완료  : {len(arrays)}개 파일, {failed}개 실패")
    if nan_files:
        print(f"  NaN 파일   : {len(nan_files)}개")
        for fn in nan_files[:3]:
            print(f"    {fn}")
    print(f"  합산 shape : {data.shape}  ({data.shape[0]//1440:.1f}일치)")
    print(f"  값 범위    : {data.min():.2f} ~ {data.max():.2f}")
    print(f"  전체 평균  : {data.mean():.2f}")
    return data


def save_model(path, pca, k, use_log):
    np.savez_compressed(path,
        mean       = pca.mean_.astype(np.float32),
        components = pca.components_.astype(np.float32),
        k          = np.array(k),
        use_log    = np.array(use_log),
        valid_idx  = np.array(VALID_IDX))
    print(f"  모델 저장  : {path}  ({os.path.getsize(path)/1024:.1f} KB)")


def compute_metrics(original, reconstructed, raw_size, compressed_size):
    """RMSE, NRMSE, CR 계산 (원본 카운트 스케일 기준)"""
    diff  = original.astype(np.float64) - reconstructed.astype(np.float64)
    rmse  = np.sqrt(np.mean(diff**2))
    nrmse = rmse / (original.mean() + 1e-12)
    cr    = raw_size / compressed_size
    return {"CR": cr, "RMSE": rmse, "NRMSE": nrmse}


def search_k(train_raw, test_raw, raw_size, use_log, k_max=30):
    train_proc = preprocess(train_raw, use_log)
    test_proc  = preprocess(test_raw,  use_log)

    k_max    = min(k_max, train_proc.shape[1], train_proc.shape[0])
    pca_full = PCA(n_components=k_max)
    pca_full.fit(train_proc)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

    print(f"\n  전처리     : {'log1p 변환 적용' if use_log else '없음 (raw count)'}")
    print(f"  test 평균  : {test_raw.mean():.2f}")
    print(f"\n{'='*65}")
    print(f"  {'k':>4} {'설명분산':>11} {'CR':>8} {'RMSE':>10} {'NRMSE':>10}")
    print(f"{'='*65}")

    for k in range(1, k_max + 1):
        mean       = pca_full.mean_
        components = pca_full.components_[:k]

        scores     = (test_proc - mean) @ components.T
        recon_proc = scores @ components + mean
        recon_raw  = np.clip(postprocess(recon_proc, use_log), 0, None).astype(np.float32)

        sc = bz2.compress(scores.astype(np.float32).tobytes(), 9)
        co = bz2.compress(components.astype(np.float32).tobytes(), 9)
        me = bz2.compress(mean.astype(np.float32).tobytes(), 9)
        cr = raw_size / (len(sc) + len(co) + len(me))

        rmse  = np.sqrt(np.mean((test_raw - recon_raw)**2))
        nrmse = rmse / (test_raw.mean() + 1e-12)

        print(f"  {k:>4} {cumvar[k-1]:>10.6f}% {cr:>8.2f}x {rmse:>10.4f} {nrmse:>10.6f}")

    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="KSEM PD PCA 압축 모델 학습")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--pd",         default="PD1")
    parser.add_argument("--k",          type=int, default=20)
    parser.add_argument("--search",     action="store_true")
    parser.add_argument("--k_max",      type=int, default=30)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_files",  type=int, default=None)
    parser.add_argument("--save",       default="pca_model.npz")
    parser.add_argument("--log",        action="store_true",
                        help="log1p 전처리 적용 (권장)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  KSEM {args.pd} PCA 압축 모델 학습")
    print(f"  유효 채널 : {len(VALID_IDX)}채널 (A:{len(A_VALID_IDX)} + B:{len(B_VALID_IDX)})")
    print(f"  log 전처리: {'✅ 적용' if args.log else '❌ 미적용'}")
    print(f"{'='*55}")

    print(f"\n[1] 파일 탐색")
    files = find_files(args.data_dir, args.pd)
    if not files:
        print(f"  [오류] 파일 없음: {args.data_dir}")
        return
    print(f"  발견: {len(files)}개 파일")
    print(f"  기간: {os.path.basename(files[0])[:8]} ~ {os.path.basename(files[-1])[:8]}")

    print(f"\n[2] 데이터 로드 (max_files={args.max_files})")
    data = load_files(files, max_files=args.max_files)

    print(f"\n[3] Train/Test 분리 (test_ratio={args.test_ratio})")
    n_test     = max(1440, int(len(data) * args.test_ratio))
    n_train    = len(data) - n_test
    train_raw  = data[:n_train]
    test_raw   = data[n_train:]
    print(f"  Train: {train_raw.shape}  ({n_train//1440:.1f}일치)")
    print(f"  Test : {test_raw.shape}   ({n_test//1440:.1f}일치)")

    if args.search:
        print(f"\n[4] k 탐색 (1 ~ {args.k_max})")
        search_k(train_raw, test_raw, test_raw.nbytes, args.log, k_max=args.k_max)
        return

    # PCA 학습
    print(f"\n[4] PCA 학습 (k={args.k})")
    train_proc = preprocess(train_raw, args.log)
    t0  = time.perf_counter()
    pca = PCA(n_components=args.k)
    pca.fit(train_proc)
    print(f"  학습 완료     : {time.perf_counter()-t0:.1f}초")
    print(f"  누적 설명분산 : {np.sum(pca.explained_variance_ratio_)*100:.6f}%")

    # 테스트 평가
    print(f"\n[5] 테스트 평가")
    test_proc  = preprocess(test_raw, args.log)
    mean, comp = pca.mean_, pca.components_

    t0     = time.perf_counter()
    scores = (test_proc - mean) @ comp.T
    sc_c   = bz2.compress(scores.astype(np.float32).tobytes(), 9)
    co_c   = bz2.compress(comp.astype(np.float32).tobytes(), 9)
    me_c   = bz2.compress(mean.astype(np.float32).tobytes(), 9)
    t_enc  = time.perf_counter() - t0
    comp_size = len(sc_c) + len(co_c) + len(me_c)

    t0         = time.perf_counter()
    recon_proc = scores @ comp + mean
    recon_raw  = np.clip(postprocess(recon_proc, args.log), 0, None).astype(np.float32)
    t_dec      = time.perf_counter() - t0

    m = compute_metrics(test_raw, recon_raw, test_raw.nbytes, comp_size)
    print(f"  압축률 (CR)  : {m['CR']:.2f}x")
    print(f"  RMSE         : {m['RMSE']:.4f}")
    print(f"  NRMSE        : {m['NRMSE']:.6f}  ({m['NRMSE']*100:.2f}%)")
    print(f"  압축 시간    : {t_enc*1000:.1f} ms")
    print(f"  복원 시간    : {t_dec*1000:.1f} ms")

    print(f"\n[6] 모델 저장")
    save_model(args.save, pca, args.k, args.log)
    print(f"\n완료!\n")


if __name__ == "__main__":
    main()