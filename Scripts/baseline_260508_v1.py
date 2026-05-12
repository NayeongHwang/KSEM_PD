"""
KSEM PD1 Raw Count 데이터 압축 알고리즘 베이스라인
=====================================================
사용법:
    python ksem_compression_baseline.py --input 20240510_PD1_Raw_Count.csv

필요 패키지:
    pip install numpy pandas
"""

import argparse
import time
import zlib
import bz2
import lzma
import gzip
import io

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────────

def delta_encode(arr: np.ndarray) -> np.ndarray:
    """시간축(행) 방향 차분 인코딩. 첫 행은 원본 그대로."""
    d = np.empty_like(arr)
    d[0] = arr[0]
    d[1:] = np.diff(arr, axis=0)
    return d


def delta_decode(d: np.ndarray) -> np.ndarray:
    """delta_encode 역변환."""
    return np.cumsum(d, axis=0)


def compute_metrics(original: np.ndarray,
                    reconstructed: np.ndarray,
                    raw_size: int,
                    compressed_size: int) -> dict:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    rmse  = np.sqrt(np.mean(diff ** 2))
    nrmse = rmse / (original.mean() + 1e-12)   # 0 나누기 방지
    cr    = raw_size / compressed_size
    return {"CR": cr, "RMSE": rmse, "NRMSE": nrmse}


# ──────────────────────────────────────────────
# 개별 알고리즘
# ──────────────────────────────────────────────

def run_bz2(data: np.ndarray, raw_size: int) -> dict:
    """bz2 lossless (float32 원본)"""
    raw = data.tobytes()
    t0  = time.perf_counter()
    c   = bz2.compress(raw, compresslevel=9)
    t_enc = time.perf_counter() - t0

    t0  = time.perf_counter()
    dec = np.frombuffer(bz2.decompress(c), dtype=np.float32).reshape(data.shape)
    t_dec = time.perf_counter() - t0

    m = compute_metrics(data, dec, raw_size, len(c))
    return {**m, "enc_ms": t_enc * 1000, "dec_ms": t_dec * 1000, "type": "lossless"}


def run_gzip(data: np.ndarray, raw_size: int) -> dict:
    """gzip lossless (float32 원본)"""
    raw = data.tobytes()
    t0  = time.perf_counter()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
        f.write(raw)
    c = buf.getvalue()
    t_enc = time.perf_counter() - t0

    t0  = time.perf_counter()
    buf2 = io.BytesIO(c)
    with gzip.GzipFile(fileobj=buf2, mode="rb") as f:
        dec = np.frombuffer(f.read(), dtype=np.float32).reshape(data.shape)
    t_dec = time.perf_counter() - t0

    m = compute_metrics(data, dec, raw_size, len(c))
    return {**m, "enc_ms": t_enc * 1000, "dec_ms": t_dec * 1000, "type": "lossless"}


def run_lzma(data: np.ndarray, raw_size: int) -> dict:
    """lzma lossless (float32 원본)"""
    raw = data.tobytes()
    t0  = time.perf_counter()
    c   = lzma.compress(raw, preset=9)
    t_enc = time.perf_counter() - t0

    t0  = time.perf_counter()
    dec = np.frombuffer(lzma.decompress(c), dtype=np.float32).reshape(data.shape)
    t_dec = time.perf_counter() - t0

    m = compute_metrics(data, dec, raw_size, len(c))
    return {**m, "enc_ms": t_enc * 1000, "dec_ms": t_dec * 1000, "type": "lossless"}


def run_delta_bz2(data: np.ndarray, raw_size: int) -> dict:
    """
    Delta encoding (시간축 차분) + bz2   [lossless]
    데이터를 *20 정수화 → 차분 → bz2 압축
    """
    # *20 정수화 (소수점이 0.05 단위이므로 완벽히 역변환 가능)
    int_data = np.round(data * 20).astype(np.int32)

    t0 = time.perf_counter()
    d  = delta_encode(int_data)
    c  = bz2.compress(d.tobytes(), compresslevel=9)
    t_enc = time.perf_counter() - t0

    t0  = time.perf_counter()
    dd  = np.frombuffer(bz2.decompress(c), dtype=np.int32).reshape(data.shape)
    dec = (delta_decode(dd) / 20.0).astype(np.float32)
    t_dec = time.perf_counter() - t0

    m = compute_metrics(data, dec, raw_size, len(c))
    return {**m, "enc_ms": t_enc * 1000, "dec_ms": t_dec * 1000, "type": "lossless"}


def run_float16_bz2(data: np.ndarray, raw_size: int) -> dict:
    """float32 → float16 양자화 + bz2   [lossy]"""
    t0 = time.perf_counter()
    c  = bz2.compress(data.astype(np.float16).tobytes(), compresslevel=9)
    t_enc = time.perf_counter() - t0

    t0  = time.perf_counter()
    dec = np.frombuffer(bz2.decompress(c), dtype=np.float16).reshape(data.shape).astype(np.float32)
    t_dec = time.perf_counter() - t0

    m = compute_metrics(data, dec, raw_size, len(c))
    return {**m, "enc_ms": t_enc * 1000, "dec_ms": t_dec * 1000, "type": "lossy"}


def run_log_float16_bz2(data: np.ndarray, raw_size: int) -> dict:
    """log1p 변환 + float16 + bz2   [lossy]"""
    t0 = time.perf_counter()
    ld = np.log1p(data).astype(np.float16)
    c  = bz2.compress(ld.tobytes(), compresslevel=9)
    t_enc = time.perf_counter() - t0

    t0  = time.perf_counter()
    ld2 = np.frombuffer(bz2.decompress(c), dtype=np.float16).reshape(data.shape)
    dec = np.expm1(ld2.astype(np.float32))
    t_dec = time.perf_counter() - t0

    m = compute_metrics(data, dec, raw_size, len(c))
    return {**m, "enc_ms": t_enc * 1000, "dec_ms": t_dec * 1000, "type": "lossy"}


def run_log_delta_bz2(data: np.ndarray, raw_size: int) -> dict:
    """
    log1p 변환 → *1000 정수화 → delta → bz2   [lossy]
    로그 스케일에서 차분하면 비율 변화량을 포착
    """
    t0 = time.perf_counter()
    log_int = np.round(np.log1p(data) * 1000).astype(np.int32)
    d  = delta_encode(log_int)
    c  = bz2.compress(d.tobytes(), compresslevel=9)
    t_enc = time.perf_counter() - t0

    t0  = time.perf_counter()
    dd  = np.frombuffer(bz2.decompress(c), dtype=np.int32).reshape(data.shape)
    dec = np.expm1(delta_decode(dd) / 1000.0).astype(np.float32)
    t_dec = time.perf_counter() - t0

    m = compute_metrics(data, dec, raw_size, len(c))
    return {**m, "enc_ms": t_enc * 1000, "dec_ms": t_dec * 1000, "type": "lossy"}


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

ALGORITHMS = {
    "bz2 (lossless)"     : run_bz2,
    "gzip (lossless)"    : run_gzip,
    "lzma (lossless)"    : run_lzma,
    "delta+bz2"          : run_delta_bz2,
    "float16+bz2"        : run_float16_bz2,
    "log+float16+bz2"    : run_log_float16_bz2,
    "log_delta+bz2"      : run_log_delta_bz2,
}


def main():
    parser = argparse.ArgumentParser(description="KSEM PD1 압축 알고리즘 베이스라인")
    parser.add_argument("--input", required=True, help="CSV 파일 경로 (예: 20240510_PD1_Raw_Count.csv)")
    parser.add_argument("--algo", default="all", help="실행할 알고리즘 (all 또는 이름 일부)")
    args = parser.parse_args()

    # ── 데이터 로드 ──
    print(f"\n[로드] {args.input}")
    df   = pd.read_csv(args.input)
    data = df.drop("Time", axis=1).values.astype(np.float32)
    raw_size = data.nbytes

    print(f"  shape   : {data.shape}  (rows=시간, cols=채널)")
    print(f"  원본 크기: {raw_size / 1024:.1f} KB  ({raw_size:,} bytes)")
    print(f"  dtype   : float32")
    print(f"  값 범위  : {data.min():.3f} ~ {data.max():.3f}")
    print(f"  0 비율   : {(data == 0).mean() * 100:.1f}%")

    # ── 알고리즘 선택 ──
    if args.algo == "all":
        selected = ALGORITHMS
    else:
        selected = {k: v for k, v in ALGORITHMS.items() if args.algo.lower() in k.lower()}
        if not selected:
            print(f"\n[오류] '{args.algo}' 에 해당하는 알고리즘 없음.")
            print(f"  사용 가능: {list(ALGORITHMS.keys())}")
            return

    # ── 실행 ──
    print(f"\n{'='*72}")
    print(f"  {'알고리즘':<22} {'유형':<10} {'CR':>6}  {'RMSE':>10}  {'NRMSE':>8}  {'enc(ms)':>8}  {'dec(ms)':>8}")
    print(f"{'='*72}")

    all_results = {}
    for name, fn in selected.items():
        try:
            r = fn(data, raw_size)
            all_results[name] = r
            print(f"  {name:<22} {r['type']:<10} {r['CR']:>6.2f}x  {r['RMSE']:>10.4f}  {r['NRMSE']:>8.4f}  {r['enc_ms']:>8.1f}  {r['dec_ms']:>8.1f}")
        except Exception as e:
            print(f"  {name:<22} [오류] {e}")

    print(f"{'='*72}")

    # ── 요약 ──
    if all_results:
        best_cr   = max(all_results, key=lambda k: all_results[k]["CR"])
        best_rmse = min(all_results, key=lambda k: all_results[k]["RMSE"])
        print(f"\n  최고 압축률 : {best_cr}  →  {all_results[best_cr]['CR']:.2f}x")
        print(f"  최저 RMSE  : {best_rmse}  →  {all_results[best_rmse]['RMSE']:.4f}")
        print()


if __name__ == "__main__":
    main()