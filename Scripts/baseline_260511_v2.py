"""
KSEM PD1 Raw Count 데이터 압축 알고리즘 베이스라인
=====================================================
유효 채널: A0~A41, A45~A56, B0~B41, B45~B56 (총 108채널, Proton only)

사용법:
    # 단일 파일
    python ksem_compression_baseline.py --input 20240510_PD1_Raw Count.csv

    # 전체 데이터 디렉토리
    python ksem_compression_baseline.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1

    # 일부만 테스트
    python ksem_compression_baseline.py --data_dir D:\\workspace\\KSEM_L0\\Raw_count\\Raw_count --pd PD1 --max_files 50

필요 패키지:
    pip install numpy pandas tqdm
"""

import argparse, time, zlib, bz2, lzma, gzip, io, os, glob
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
VALID_IDX   = A_VALID_IDX + B_VALID_IDX  # 총 108채널


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

def load_single(path: str) -> np.ndarray:
    """단일 CSV → 108채널 float32"""
    df       = pd.read_csv(path)
    all_data = df.drop("Time", axis=1).values.astype(np.float32)
    arr      = all_data[:, VALID_IDX]
    arr      = np.nan_to_num(arr, nan=0.0)
    return arr


def find_files(data_dir: str, pd_name: str) -> list:
    pattern = os.path.join(data_dir, "*", f"*_{pd_name}_Raw Count.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(data_dir, "*", f"*_{pd_name}_Raw_Count.csv")
        files   = sorted(glob.glob(pattern))
    return files


def load_directory(data_dir: str, pd_name: str, max_files=None) -> np.ndarray:
    files = find_files(data_dir, pd_name)
    if not files:
        raise FileNotFoundError(f"파일 없음: {data_dir}")
    if max_files:
        files = files[:max_files]

    print(f"  발견: {len(files)}개 파일")
    print(f"  기간: {os.path.basename(files[0])[:8]} ~ {os.path.basename(files[-1])[:8]}")

    arrays, failed, nan_count = [], 0, 0
    iterator = tqdm(files, desc="파일 로드") if USE_TQDM else files

    for f in iterator:
        try:
            df       = pd.read_csv(f)
            all_data = df.drop("Time", axis=1).values.astype(np.float32)
            if all_data.shape[1] != 256:
                continue
            arr = all_data[:, VALID_IDX]
            if np.isnan(arr).sum() > 0:
                nan_count += 1
                arr = arr[~np.isnan(arr).any(axis=1)]
                arr = np.nan_to_num(arr, nan=0.0)
                if len(arr) == 0:
                    continue
            arrays.append(arr)
        except Exception:
            failed += 1

    data = np.vstack(arrays)
    print(f"\n  로드 완료  : {len(arrays)}개 파일, {failed}개 실패, NaN파일 {nan_count}개")
    print(f"  합산 shape : {data.shape}  ({data.shape[0]//1440:.1f}일치)")
    print(f"  유효 채널  : {data.shape[1]}채널")
    print(f"  값 범위    : {data.min():.2f} ~ {data.max():.2f}")
    print(f"  전체 평균  : {data.mean():.2f}")
    return data


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def delta_encode(arr: np.ndarray) -> np.ndarray:
    d = np.empty_like(arr)
    d[0]  = arr[0]
    d[1:] = np.diff(arr, axis=0)
    return d

def delta_decode(d: np.ndarray) -> np.ndarray:
    return np.cumsum(d, axis=0)

def compute_metrics(original, reconstructed, raw_size, compressed_size):
    diff  = original.astype(np.float64) - reconstructed.astype(np.float64)
    rmse  = np.sqrt(np.mean(diff**2))
    nrmse = rmse / (original.mean() + 1e-12)
    cr    = raw_size / compressed_size
    return {"CR": cr, "RMSE": rmse, "NRMSE": nrmse}


# ──────────────────────────────────────────────
# 알고리즘
# ──────────────────────────────────────────────

def run_bz2(data, raw_size):
    raw = data.tobytes()
    t0  = time.perf_counter(); c = bz2.compress(raw, 9); t_enc = time.perf_counter()-t0
    t0  = time.perf_counter()
    dec = np.frombuffer(bz2.decompress(c), dtype=np.float32).reshape(data.shape)
    t_dec = time.perf_counter()-t0
    return {**compute_metrics(data, dec, raw_size, len(c)),
            "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossless"}


def run_gzip(data, raw_size):
    raw = data.tobytes()
    t0  = time.perf_counter()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
        f.write(raw)
    c = buf.getvalue(); t_enc = time.perf_counter()-t0
    t0  = time.perf_counter()
    buf2 = io.BytesIO(c)
    with gzip.GzipFile(fileobj=buf2, mode="rb") as f:
        dec = np.frombuffer(f.read(), dtype=np.float32).reshape(data.shape)
    t_dec = time.perf_counter()-t0
    return {**compute_metrics(data, dec, raw_size, len(c)),
            "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossless"}


def run_lzma(data, raw_size):
    raw = data.tobytes()
    t0  = time.perf_counter(); c = lzma.compress(raw, preset=9); t_enc = time.perf_counter()-t0
    t0  = time.perf_counter()
    dec = np.frombuffer(lzma.decompress(c), dtype=np.float32).reshape(data.shape)
    t_dec = time.perf_counter()-t0
    return {**compute_metrics(data, dec, raw_size, len(c)),
            "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossless"}


def run_delta_bz2(data, raw_size):
    int_data = np.round(data * 20).astype(np.int32)
    t0  = time.perf_counter()
    d   = delta_encode(int_data); c = bz2.compress(d.tobytes(), 9); t_enc = time.perf_counter()-t0
    t0  = time.perf_counter()
    dd  = np.frombuffer(bz2.decompress(c), dtype=np.int32).reshape(data.shape)
    dec = (delta_decode(dd) / 20.0).astype(np.float32); t_dec = time.perf_counter()-t0
    return {**compute_metrics(data, dec, raw_size, len(c)),
            "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossless"}


def run_float16_bz2(data, raw_size):
    t0  = time.perf_counter(); c = bz2.compress(data.astype(np.float16).tobytes(), 9); t_enc = time.perf_counter()-t0
    t0  = time.perf_counter()
    dec = np.frombuffer(bz2.decompress(c), dtype=np.float16).reshape(data.shape).astype(np.float32)
    t_dec = time.perf_counter()-t0
    return {**compute_metrics(data, dec, raw_size, len(c)),
            "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossy"}


def run_log_float16_bz2(data, raw_size):
    t0  = time.perf_counter()
    ld  = np.log1p(data).astype(np.float16); c = bz2.compress(ld.tobytes(), 9); t_enc = time.perf_counter()-t0
    t0  = time.perf_counter()
    ld2 = np.frombuffer(bz2.decompress(c), dtype=np.float16).reshape(data.shape)
    dec = np.expm1(ld2.astype(np.float32)); t_dec = time.perf_counter()-t0
    return {**compute_metrics(data, dec, raw_size, len(c)),
            "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossy"}


def run_log_delta_bz2(data, raw_size):
    t0      = time.perf_counter()
    log_int = np.round(np.log1p(data) * 1000).astype(np.int32)
    d       = delta_encode(log_int); c = bz2.compress(d.tobytes(), 9); t_enc = time.perf_counter()-t0
    t0  = time.perf_counter()
    dd  = np.frombuffer(bz2.decompress(c), dtype=np.int32).reshape(data.shape)
    dec = np.expm1(delta_decode(dd) / 1000.0).astype(np.float32); t_dec = time.perf_counter()-t0
    return {**compute_metrics(data, dec, raw_size, len(c)),
            "enc_ms": t_enc*1000, "dec_ms": t_dec*1000, "type": "lossy"}


ALGORITHMS = {
    "bz2"           : run_bz2,
    "gzip"          : run_gzip,
    "lzma"          : run_lzma,
    "delta+bz2"     : run_delta_bz2,
    "float16+bz2"   : run_float16_bz2,
    "log+f16+bz2"   : run_log_float16_bz2,
    "log_delta+bz2" : run_log_delta_bz2,
}


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KSEM PD 압축 알고리즘 베이스라인")
    parser.add_argument("--input",      default=None, help="단일 CSV 파일 경로")
    parser.add_argument("--data_dir",   default=None, help="전체 데이터 디렉토리")
    parser.add_argument("--pd",         default="PD1", help="PD 종류 (PD1/PD2/PD3)")
    parser.add_argument("--max_files",  type=int, default=None, help="최대 파일 수 (테스트용)")
    parser.add_argument("--algo",       default="all", help="알고리즘 선택 (all 또는 이름 일부)")
    args = parser.parse_args()

    if args.input is None and args.data_dir is None:
        parser.error("--input 또는 --data_dir 중 하나 필요")

    # ── 데이터 로드 ──
    print(f"\n{'='*72}")
    print(f"  KSEM {args.pd} 압축 알고리즘 베이스라인")
    print(f"  유효 채널: {len(VALID_IDX)}채널 (A:{len(A_VALID_IDX)} + B:{len(B_VALID_IDX)})")
    print(f"{'='*72}")

    print(f"\n[데이터 로드]")
    if args.input:
        print(f"  파일: {args.input}")
        data = load_single(args.input)
        print(f"  shape: {data.shape}")
        print(f"  값 범위: {data.min():.2f} ~ {data.max():.2f}")
        print(f"  전체 평균: {data.mean():.2f}")
    else:
        data = load_directory(args.data_dir, args.pd, max_files=args.max_files)

    raw_size = data.nbytes
    print(f"  원본 크기: {raw_size/1024/1024:.1f} MB")

    # ── 알고리즘 선택 ──
    if args.algo == "all":
        selected = ALGORITHMS
    else:
        selected = {k: v for k, v in ALGORITHMS.items() if args.algo.lower() in k.lower()}
        if not selected:
            print(f"\n[오류] '{args.algo}' 없음. 사용 가능: {list(ALGORITHMS.keys())}")
            return

    # ── 실행 ──
    print(f"\n{'='*75}")
    print(f"  {'알고리즘':<16} {'유형':<10} {'CR':>7} {'RMSE':>10} {'NRMSE':>10} {'enc(ms)':>9} {'dec(ms)':>9}")
    print(f"{'='*75}")

    all_results = {}
    for name, fn in selected.items():
        try:
            r = fn(data, raw_size)
            all_results[name] = r
            print(f"  {name:<16} {r['type']:<10} {r['CR']:>7.2f}x {r['RMSE']:>10.4f} {r['NRMSE']:>10.6f} {r['enc_ms']:>9.1f} {r['dec_ms']:>9.1f}")
        except Exception as e:
            print(f"  {name:<16} [오류] {e}")

    print(f"{'='*75}")

    if all_results:
        best_cr   = max(all_results, key=lambda k: all_results[k]["CR"])
        best_rmse = min(all_results, key=lambda k: all_results[k]["RMSE"])
        print(f"\n  최고 압축률: {best_cr}  →  {all_results[best_cr]['CR']:.2f}x")
        print(f"  최저 RMSE : {best_rmse}  →  {all_results[best_rmse]['RMSE']:.4f}")
        print()


if __name__ == "__main__":
    main()