"""
KSEM PD1 Log Map Compression Baseline
--------------------------------------
NASA standard log compression (8-bit quantization) 적용 및
기존 베이스라인(raw+bz2, log1p+f16+bz2)과 비교.

사용법:
    python ksem_logmap_baseline.py --data "path/to/YYYYMMDD_PD1_Raw Count.csv"
    python ksem_logmap_baseline.py --data "path/to/YYYYMMDD_PD1_Raw Count.csv" --plot
"""

import argparse
import bz2
import os

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 1. NASA Log Compression Map (index 0~255 → count)
# ──────────────────────────────────────────────
LOG_MAP = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
    128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
    256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496,
    512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992,
    1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984,
    2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968,
    4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936,
    8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872,
    16384, 17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624, 27648, 28672, 29696, 30720, 31744,
    32768, 34816, 36864, 38912, 40960, 43008, 45056, 47104, 49152, 51200, 53248, 55296, 57344, 59392, 61440, 63488,
    65536, 69632, 73728, 77824, 81920, 86016, 90112, 94208, 98304, 102400, 106496, 110592, 114688, 118784, 122880, 126976,
    131072, 139264, 147456, 155648, 163840, 172032, 180224, 188416, 196608, 204800, 212992, 221184, 229376, 237568, 245760, 253952,
    262144, 278528, 294912, 311296, 327680, 344064, 360448, 376832, 393216, 409600, 425984, 442368, 458752, 475136, 491520, 507904,
], dtype=np.int32)


# ──────────────────────────────────────────────
# 2. 인코딩 / 디코딩
# ──────────────────────────────────────────────
def logmap_encode(raw: np.ndarray) -> np.ndarray:
    """count 값 → uint8 index (lookup table 기반)"""
    idx = np.searchsorted(LOG_MAP, raw, side="right") - 1
    return np.clip(idx, 0, 255).astype(np.uint8)


def logmap_decode(idx: np.ndarray) -> np.ndarray:
    """uint8 index → count 값 (복원)"""
    return LOG_MAP[idx].astype(np.float32)


# ──────────────────────────────────────────────
# 3. 채널 필터링 (108채널: A0~A56 + B0~B56, 42/43/44 제외)
# ──────────────────────────────────────────────
EXCLUDE_IDX = {42, 43, 44}


def get_valid_columns(df: pd.DataFrame) -> list:
    valid = []
    for col in df.columns:
        if col == "Time":
            continue
        prefix = col[0]
        try:
            num = int(col.split("(")[0][1:])
        except ValueError:
            continue
        if prefix in ("A", "B") and 0 <= num <= 56 and num not in EXCLUDE_IDX:
            valid.append(col)
    return valid


# ──────────────────────────────────────────────
# 4. 평가 지표
# ──────────────────────────────────────────────
def compute_metrics(raw: np.ndarray, recon: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((recon - raw) ** 2)))

    log_orig  = np.log1p(raw)
    log_recon = np.log1p(recon)
    log_range = float(log_orig.max() - log_orig.min())
    log_nrmse = float(np.sqrt(np.mean((log_orig - log_recon) ** 2)) / log_range) if log_range > 0 else 0.0

    mask = raw > 0
    rel_err = np.abs(recon - raw)[mask] / raw[mask]
    mean_rel_err = float(rel_err.mean()) if mask.any() else 0.0

    return {"RMSE": rmse, "log_NRMSE": log_nrmse, "mean_rel_err_pct": mean_rel_err * 100}


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / compressed_bytes


# ──────────────────────────────────────────────
# 5. 각 방법별 압축/복원 + 지표 계산
# ──────────────────────────────────────────────
def run_baselines(raw: np.ndarray) -> list:
    results = []
    raw_bytes = raw.nbytes  # float32 기준

    # ── (1) raw + bz2 (무손실)
    compressed = bz2.compress(raw.tobytes())
    cr = compression_ratio(raw_bytes, len(compressed))
    results.append({
        "method": "raw + bz2",
        "CR": cr,
        "log_NRMSE": 0.0,
        "RMSE": 0.0,
        "mean_rel_err_pct": 0.0,
        "note": "무손실",
    })

    # ── (2) log_map only (uint8, bz2 없음)
    enc = logmap_encode(raw)
    dec = logmap_decode(enc)
    m = compute_metrics(raw, dec)
    cr = compression_ratio(raw_bytes, enc.nbytes)
    results.append({"method": "log_map (uint8)", "CR": cr, **m, "note": "온보드 단독"})

    # ── (3) log_map + bz2
    compressed = bz2.compress(enc.tobytes())
    cr = compression_ratio(raw_bytes, len(compressed))
    results.append({"method": "log_map + bz2", "CR": cr, **m, "note": ""})

    # ── (4) log1p + float16 + bz2
    lf16 = np.log1p(raw).astype(np.float16)
    lf16_dec = np.expm1(lf16.astype(np.float32))
    m2 = compute_metrics(raw, lf16_dec)
    compressed = bz2.compress(lf16.tobytes())
    cr = compression_ratio(raw_bytes, len(compressed))
    results.append({"method": "log1p + f16 + bz2", "CR": cr, **m2, "note": ""})

    return results


# ──────────────────────────────────────────────
# 6. 결과 출력
# ──────────────────────────────────────────────
def print_results(results: list, data_info: str = ""):
    print(f"\n{'='*60}")
    print(f"  KSEM PD1 압축 베이스라인 비교  {data_info}")
    print(f"{'='*60}")
    header = f"{'방법':<22} {'CR':>7}  {'log-NRMSE':>11}  {'RMSE':>8}  비고"
    print(header)
    print("-" * 60)
    for r in results:
        print(
            f"{r['method']:<22} {r['CR']:>7.2f}x"
            f"  {r['log_NRMSE']:>11.6f}"
            f"  {r['RMSE']:>8.2f}"
            f"  {r['note']}"
        )
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 7. (선택) 시각화
# ──────────────────────────────────────────────
def plot_results(raw: np.ndarray, results: list, save_path: str = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[경고] matplotlib 없음 — 플롯 생략")
        return

    enc = logmap_encode(raw)
    dec_logmap = logmap_decode(enc)

    lf16 = np.log1p(raw).astype(np.float16)
    dec_lf16 = np.expm1(lf16.astype(np.float32))

    # 채널 0 시계열 비교
    ch = 0
    t = np.arange(raw.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("KSEM PD1 — 압축 방법별 복원 비교 (채널 0)", fontsize=12)

    axes[0].plot(t, raw[:, ch], label="원본", lw=1)
    axes[0].plot(t, dec_logmap[:, ch], "--", label="log_map 복원", lw=1)
    axes[0].set_ylabel("Count (CPS)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, np.abs(dec_logmap[:, ch] - raw[:, ch]), label="|오차| log_map", lw=1, color="orange")
    axes[1].plot(t, np.abs(dec_lf16[:, ch]  - raw[:, ch]), "--", label="|오차| log1p+f16", lw=1, color="green")
    axes[1].set_ylabel("|오차|")
    axes[1].legend(fontsize=8)

    # CR 막대 그래프
    methods = [r["method"] for r in results]
    crs     = [r["CR"]     for r in results]
    axes[2].bar(methods, crs, color=["steelblue", "orange", "tomato", "green"])
    axes[2].set_ylabel("Compression Ratio (x)")
    axes[2].tick_params(axis="x", labelsize=8)
    for i, v in enumerate(crs):
        axes[2].text(i, v + 0.3, f"{v:.1f}x", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[저장됨] {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────
# 8. main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KSEM PD1 Log Map 압축 베이스라인")
    parser.add_argument("--data", required=True, help="CSV 파일 경로")
    parser.add_argument("--plot", action="store_true", help="결과 시각화")
    parser.add_argument("--save_plot", default=None, help="플롯 저장 경로 (.png)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"파일 없음: {args.data}")

    # 데이터 로드
    print(f"[로드] {args.data}")
    df = pd.read_csv(args.data)
    valid_cols = get_valid_columns(df)
    print(f"[채널] {len(valid_cols)}개 (A:{sum(1 for c in valid_cols if c.startswith('A'))}, "
          f"B:{sum(1 for c in valid_cols if c.startswith('B'))})")

    raw = df[valid_cols].values.astype(np.float32)
    print(f"[shape] {raw.shape},  count 범위: {raw.min():.0f} ~ {raw.max():.0f}")

    # 베이스라인 실행
    results = run_baselines(raw)
    fname = os.path.basename(args.data)
    print_results(results, data_info=f"({fname})")

    # 시각화
    if args.plot or args.save_plot:
        plot_results(raw, results, save_path=args.save_plot)


if __name__ == "__main__":
    main()
