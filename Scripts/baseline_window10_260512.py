"""
KSEM PD1 Raw Count 데이터 압축 알고리즘 베이스라인
=====================================================
유효 채널: A0~A41, A45~A56, B0~B41, B45~B56 (총 108채널, Proton only)

사용법:
    # 전체 데이터 (한 덩어리)
    python ksem_compression_baseline.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1

    # window 단위 압축 (딥러닝과 공평한 비교)
    python ksem_compression_baseline.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --window 10

    # window 탐색 (5, 10, 30분 비교)
    python ksem_compression_baseline.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --search_window

    # 단일 파일
    python ksem_compression_baseline.py --input "20240510_PD1_Raw Count.csv"

필요 패키지:
    pip install numpy pandas tqdm
"""

import argparse, time, zlib, bz2, lzma, gzip, io, os, glob, random
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ── 유효 채널 정의 ──
A_VALID_IDX = [i for i in range(57) if i not in (42, 43, 44)]
B_VALID_IDX = [128 + i for i in range(57) if i not in (42, 43, 44)]
VALID_IDX   = A_VALID_IDX + B_VALID_IDX  # 108채널


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
    """단일 파일 → 108채널 float32"""
    df       = pd.read_csv(f)
    all_data = df.drop("Time", axis=1).values.astype(np.float32)
    if all_data.shape[1] != 256:
        return None
    arr = all_data[:, VALID_IDX]
    if np.isnan(arr).sum() > 0:
        arr = arr[~np.isnan(arr).any(axis=1)]
        arr = np.nan_to_num(arr, nan=0.0)
    return arr if len(arr) > 0 else None


def load_all(data_dir, pd_name, max_files=None):
    """전체 파일 로드 → 하나의 배열로 합침"""
    files = find_files(data_dir, pd_name)
    if not files:
        raise FileNotFoundError(f"파일 없음: {data_dir}")
    if max_files:
        files = files[:max_files]

    print(f"  발견: {len(files)}개 파일")
    print(f"  기간: {os.path.basename(files[0])[:8]} ~ {os.path.basename(files[-1])[:8]}")

    arrays, failed, nan_cnt = [], 0, 0
    iterator = tqdm(files, desc="파일 로드") if USE_TQDM else files

    for f in iterator:
        try:
            arr = load_file(f)
            if arr is None:
                continue
            if np.isnan(arr).sum() > 0:
                nan_cnt += 1
            arrays.append(arr)
        except Exception:
            failed += 1

    data = np.vstack(arrays)
    print(f"\n  로드 완료  : {len(arrays)}개 파일, {failed}개 실패, NaN파일 {nan_cnt}개")
    print(f"  합산 shape : {data.shape}  ({data.shape[0]//1440:.1f}일치)")
    print(f"  값 범위    : {data.min():.2f} ~ {data.max():.2f}")
    print(f"  전체 평균  : {data.mean():.2f}")
    return data, files


def make_windows_from_files(files, window_size):
    """
    파일(날짜) 단위로 window 생성.
    날짜 경계를 넘는 window 없음 → 딥러닝과 동일한 조건.
    """
    all_windows = []
    iterator    = tqdm(files, desc=f"window 생성 ({window_size}분)") if USE_TQDM else files

    for f in iterator:
        try:
            arr = load_file(f)
            if arr is None:
                continue
            n_win = len(arr) // window_size
            if n_win == 0:
                continue
            arr_cut = arr[:n_win * window_size]
            windows = arr_cut.reshape(n_win, window_size, len(VALID_IDX))
            all_windows.append(windows)
        except Exception:
            pass

    if not all_windows:
        return np.zeros((0, window_size, len(VALID_IDX)), dtype=np.float32)
    return np.concatenate(all_windows, axis=0)


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def delta_encode(arr):
    d = np.empty_like(arr)
    d[0]  = arr[0]
    d[1:] = np.diff(arr, axis=0)
    return d

def delta_decode(d):
    return np.cumsum(d, axis=0)

def compute_metrics(original, reconstructed, raw_size, compressed_size):
    diff  = original.astype(np.float64) - reconstructed.astype(np.float64)
    rmse  = np.sqrt(np.mean(diff**2))
    nrmse = rmse / (original.mean() + 1e-12)
    cr    = raw_size / compressed_size
    # weighted RMSE (log1p 기반 가중치)
    log_orig = np.log1p(np.abs(original)).astype(np.float64)
    weights  = log_orig / (log_orig.mean() + 1e-12)
    w_rmse   = np.sqrt(np.mean(weights * diff**2))
    return {"CR": cr, "RMSE": rmse, "NRMSE": nrmse, "w_RMSE": w_rmse}


# ──────────────────────────────────────────────
# 알고리즘 (window 단위 또는 전체)
# ──────────────────────────────────────────────

def run_algorithm(name, data, raw_size):
    """
    data: (N, 108) 또는 (N_win, window, 108)
    알고리즘별로 압축/복원 수행
    """
    flat = data.reshape(-1, data.shape[-1])  # window면 펼침

    if name == "bz2":
        t0 = time.perf_counter()
        c  = bz2.compress(flat.tobytes(), 9)
        t_enc = time.perf_counter() - t0
        t0  = time.perf_counter()
        dec = np.frombuffer(bz2.decompress(c), dtype=np.float32).reshape(flat.shape)
        t_dec = time.perf_counter() - t0
        return {**compute_metrics(flat, dec, raw_size, len(c)),
                "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossless"}

    elif name == "gzip":
        t0  = time.perf_counter()
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
            f.write(flat.tobytes())
        c = buf.getvalue(); t_enc = time.perf_counter()-t0
        t0  = time.perf_counter()
        buf2 = io.BytesIO(c)
        with gzip.GzipFile(fileobj=buf2, mode="rb") as f:
            dec = np.frombuffer(f.read(), dtype=np.float32).reshape(flat.shape)
        t_dec = time.perf_counter()-t0
        return {**compute_metrics(flat, dec, raw_size, len(c)),
                "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossless"}

    elif name == "lzma":
        t0  = time.perf_counter()
        c   = lzma.compress(flat.tobytes(), preset=9); t_enc = time.perf_counter()-t0
        t0  = time.perf_counter()
        dec = np.frombuffer(lzma.decompress(c), dtype=np.float32).reshape(flat.shape)
        t_dec = time.perf_counter()-t0
        return {**compute_metrics(flat, dec, raw_size, len(c)),
                "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossless"}

    elif name == "delta+bz2":
        int_data = np.round(flat * 20).astype(np.int32)
        t0  = time.perf_counter()
        d   = delta_encode(int_data); c = bz2.compress(d.tobytes(), 9); t_enc = time.perf_counter()-t0
        t0  = time.perf_counter()
        dd  = np.frombuffer(bz2.decompress(c), dtype=np.int32).reshape(flat.shape)
        dec = (delta_decode(dd) / 20.0).astype(np.float32); t_dec = time.perf_counter()-t0
        return {**compute_metrics(flat, dec, raw_size, len(c)),
                "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossless"}

    elif name == "log+f16+bz2":
        t0  = time.perf_counter()
        ld  = np.log1p(flat).astype(np.float16)
        c   = bz2.compress(ld.tobytes(), 9); t_enc = time.perf_counter()-t0
        t0  = time.perf_counter()
        ld2 = np.frombuffer(bz2.decompress(c), dtype=np.float16).reshape(flat.shape)
        dec = np.expm1(ld2.astype(np.float32)); t_dec = time.perf_counter()-t0
        return {**compute_metrics(flat, dec, raw_size, len(c)),
                "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossy"}

    elif name == "log_delta+bz2":
        t0      = time.perf_counter()
        log_int = np.round(np.log1p(flat) * 1000).astype(np.int32)
        d       = delta_encode(log_int); c = bz2.compress(d.tobytes(), 9); t_enc = time.perf_counter()-t0
        t0  = time.perf_counter()
        dd  = np.frombuffer(bz2.decompress(c), dtype=np.int32).reshape(flat.shape)
        dec = np.expm1(delta_decode(dd) / 1000.0).astype(np.float32); t_dec = time.perf_counter()-t0
        return {**compute_metrics(flat, dec, raw_size, len(c)),
                "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossy"}


def run_window_mode(files, window_size, algo_names):
    """
    window 단위 압축: 딥러닝과 동일한 조건.
    각 window를 독립적으로 압축 → 압축 크기 합산.
    """
    print(f"\n  window={window_size}분 단위로 각각 압축 (딥러닝과 동일 조건)")
    windows = make_windows_from_files(files, window_size)  # (N_win, window, 108)
    print(f"  windows shape: {windows.shape}")

    raw_size = windows.nbytes
    results  = {}

    for name in algo_names:
        total_comp = 0
        t_enc_total = 0
        t_dec_total = 0
        all_orig  = []
        all_recon = []

        for i in range(len(windows)):
            w = windows[i]  # (window, 108)
            flat = w.reshape(1, -1)  # 1 × (window×108)

            if name == "log+f16+bz2":
                t0  = time.perf_counter()
                ld  = np.log1p(w).astype(np.float16)
                c   = bz2.compress(ld.tobytes(), 9)
                t_enc_total += time.perf_counter() - t0
                t0  = time.perf_counter()
                ld2 = np.frombuffer(bz2.decompress(c), dtype=np.float16).reshape(w.shape)
                dec = np.expm1(ld2.astype(np.float32))
                t_dec_total += time.perf_counter() - t0
            elif name == "bz2":
                t0  = time.perf_counter()
                c   = bz2.compress(w.tobytes(), 9)
                t_enc_total += time.perf_counter() - t0
                t0  = time.perf_counter()
                dec = np.frombuffer(bz2.decompress(c), dtype=np.float32).reshape(w.shape)
                t_dec_total += time.perf_counter() - t0
            elif name == "delta+bz2":
                int_w = np.round(w * 20).astype(np.int32)
                t0    = time.perf_counter()
                d     = delta_encode(int_w); c = bz2.compress(d.tobytes(), 9)
                t_enc_total += time.perf_counter() - t0
                t0    = time.perf_counter()
                dd    = np.frombuffer(bz2.decompress(c), dtype=np.int32).reshape(w.shape)
                dec   = (delta_decode(dd) / 20.0).astype(np.float32)
                t_dec_total += time.perf_counter() - t0
            elif name == "log_delta+bz2":
                t0      = time.perf_counter()
                log_int = np.round(np.log1p(w) * 1000).astype(np.int32)
                d       = delta_encode(log_int); c = bz2.compress(d.tobytes(), 9)
                t_enc_total += time.perf_counter() - t0
                t0  = time.perf_counter()
                dd  = np.frombuffer(bz2.decompress(c), dtype=np.int32).reshape(w.shape)
                dec = np.expm1(delta_decode(dd) / 1000.0).astype(np.float32)
                t_dec_total += time.perf_counter() - t0
            else:
                continue

            total_comp += len(c)
            all_orig.append(w)
            all_recon.append(dec)

        if not all_orig:
            continue

        orig_arr  = np.concatenate([x.reshape(-1) for x in all_orig])
        recon_arr = np.concatenate([x.reshape(-1) for x in all_recon])
        m = compute_metrics(orig_arr, recon_arr, raw_size, total_comp)
        results[name] = {**m,
                         "enc_ms": t_enc_total*1000,
                         "dec_ms": t_dec_total*1000}
    return results


ALGO_NAMES = ["bz2", "delta+bz2", "log+f16+bz2", "log_delta+bz2"]


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KSEM PD 압축 알고리즘 베이스라인")
    parser.add_argument("--input",         default=None)
    parser.add_argument("--data_dir",      default=None)
    parser.add_argument("--pd",            default="PD1")
    parser.add_argument("--max_files",     type=int, default=None)
    parser.add_argument("--window",        type=int, default=None,
                        help="window 크기 (분). 지정 시 window 단위 압축 (딥러닝과 동일 조건)")
    parser.add_argument("--search_window", action="store_true",
                        help="window 탐색 (5, 10, 30분 비교)")
    parser.add_argument("--algo",          default="all")
    args = parser.parse_args()

    if args.input is None and args.data_dir is None:
        parser.error("--input 또는 --data_dir 중 하나 필요")

    print(f"\n{'='*78}")
    print(f"  KSEM {args.pd} 압축 알고리즘 베이스라인")
    print(f"  유효 채널: {len(VALID_IDX)}채널 (A:{len(A_VALID_IDX)} + B:{len(B_VALID_IDX)})")
    print(f"{'='*78}")

    # 데이터 로드
    print(f"\n[데이터 로드]")
    if args.input:
        arr  = load_file(args.input)
        data = arr
        files = None
        print(f"  shape: {data.shape}")
    else:
        data, files = load_all(args.data_dir, args.pd, max_files=args.max_files)

    raw_size = data.nbytes
    print(f"  원본 크기: {raw_size/1024/1024:.1f} MB")

    algo_names = ALGO_NAMES if args.algo == "all" else \
                 [n for n in ALGO_NAMES if args.algo.lower() in n.lower()]

    # ── window 탐색 모드 ──
    if args.search_window and files:
        search_windows = [5, 10, 30]
        print(f"\n[Window 탐색: {search_windows}분]")
        print(f"  전체(bulk) 결과와 비교하기 위해 bulk도 함께 실행")

        # bulk 결과 먼저
        print(f"\n  --- bulk (전체 한 덩어리) ---")
        print(f"\n{'='*85}")
        print(f"  {'모드':<12} {'알고리즘':<16} {'CR':>7} {'RMSE':>10} {'NRMSE':>10} {'w_RMSE':>10}")
        print(f"{'='*85}")

        for name in algo_names:
            try:
                r = run_algorithm(name, data, raw_size)
                print(f"  {'bulk':<12} {name:<16} {r['CR']:>7.2f}x {r['RMSE']:>10.4f} "
                      f"{r['NRMSE']:>10.6f} {r['w_RMSE']:>10.4f}")
            except Exception as e:
                print(f"  {'bulk':<12} {name:<16} [오류] {e}")

        for w in search_windows:
            print(f"\n  --- window={w}분 ---")
            results = run_window_mode(files, w, algo_names)
            for name, r in results.items():
                print(f"  {'win='+str(w)+'분':<12} {name:<16} {r['CR']:>7.2f}x {r['RMSE']:>10.4f} "
                      f"{r['NRMSE']:>10.6f} {r['w_RMSE']:>10.4f}")

        print(f"\n{'='*85}")
        return

    # ── window 단위 압축 ──
    if args.window and files:
        print(f"\n[window={args.window}분 단위 압축]")
        results = run_window_mode(files, args.window, algo_names)
        print(f"\n{'='*78}")
        print(f"  {'알고리즘':<16} {'CR':>7} {'RMSE':>10} {'NRMSE':>10} {'w_RMSE':>10} {'enc(ms)':>10} {'dec(ms)':>10}")
        print(f"{'='*78}")
        for name, r in results.items():
            print(f"  {name:<16} {r['CR']:>7.2f}x {r['RMSE']:>10.4f} "
                  f"{r['NRMSE']:>10.6f} {r['w_RMSE']:>10.4f} "
                  f"{r['enc_ms']:>10.1f} {r['dec_ms']:>10.1f}")
        print(f"{'='*78}")
        return

    # ── 전체 한 덩어리 압축 (기본) ──
    print(f"\n{'='*78}")
    print(f"  {'알고리즘':<16} {'유형':<10} {'CR':>7} {'RMSE':>10} {'NRMSE':>10} {'w_RMSE':>10} {'enc(ms)':>9} {'dec(ms)':>9}")
    print(f"{'='*78}")

    all_results = {}
    for name in algo_names:
        try:
            r = run_algorithm(name, data, raw_size)
            all_results[name] = r
            print(f"  {name:<16} {r.get('type',''):<10} {r['CR']:>7.2f}x "
                  f"{r['RMSE']:>10.4f} {r['NRMSE']:>10.6f} {r['w_RMSE']:>10.4f} "
                  f"{r['enc_ms']:>9.1f} {r['dec_ms']:>9.1f}")
        except Exception as e:
            print(f"  {name:<16} [오류] {e}")

    print(f"{'='*78}")


if __name__ == "__main__":
    main()