"""
KSEM PD1 데이터에서 폭풍 날 / 조용한 날 분류
=============================================
사용법:
    python ksem_analyze_storm_days.py --data_dir D:\\workspace\\KSEM_L0_PD1\\Raw_count\\Raw_count --pd PD1

필요 패키지:
    pip install numpy pandas tqdm
"""

import argparse, os, glob
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ── 유효 채널 ──
A_VALID_IDX = [i for i in range(57) if i not in (42, 43, 44)]
B_VALID_IDX = [128 + i for i in range(57) if i not in (42, 43, 44)]
VALID_IDX   = A_VALID_IDX + B_VALID_IDX


def find_files(data_dir, pd_name):
    pattern = os.path.join(data_dir, "*", f"*_{pd_name}_Raw Count.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(data_dir, "*", f"*_{pd_name}_Raw_Count.csv")
        files   = sorted(glob.glob(pattern))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--pd",       default="PD1")
    args = parser.parse_args()

    files = find_files(args.data_dir, args.pd)
    print(f"\n총 파일 수: {len(files)}개")

    results = []
    iterator = tqdm(files, desc="분석 중") if USE_TQDM else files

    for f in iterator:
        try:
            df       = pd.read_csv(f)
            all_data = df.drop("Time", axis=1).values.astype(np.float32)
            if all_data.shape[1] != 256:
                continue
            arr      = all_data[:, VALID_IDX]
            arr      = np.nan_to_num(arr, nan=0.0)
            mean_val = arr.mean()
            max_val  = arr.max()
            date     = os.path.basename(f)[:8]
            results.append({"date": date, "mean": mean_val, "max": max_val, "file": f})
        except Exception:
            pass

    df_res = pd.DataFrame(results)

    # 분류 기준 (mean 기준)
    quiet  = df_res[df_res["mean"] < 50]
    active = df_res[(df_res["mean"] >= 50) & (df_res["mean"] < 200)]
    storm  = df_res[df_res["mean"] >= 200]

    print(f"\n{'='*55}")
    print(f"  날짜별 평균 카운트 분포")
    print(f"{'='*55}")
    print(f"  조용한 날 (mean < 50)  : {len(quiet):>5}일  ({len(quiet)/len(df_res)*100:.1f}%)")
    print(f"  보통 날  (50~200)      : {len(active):>5}일  ({len(active)/len(df_res)*100:.1f}%)")
    print(f"  폭풍 날  (mean ≥ 200)  : {len(storm):>5}일  ({len(storm)/len(df_res)*100:.1f}%)")
    print(f"{'='*55}")
    print(f"  전체 평균: {df_res['mean'].mean():.2f}")
    print(f"  전체 최대: {df_res['max'].max():.2f}")

    if len(storm) > 0:
        print(f"\n  폭풍 날 목록 (상위 20개):")
        top_storm = storm.nlargest(20, "mean")
        for _, row in top_storm.iterrows():
            print(f"    {row['date']}  mean={row['mean']:.2f}  max={row['max']:.2f}")

    if len(active) > 0:
        print(f"\n  보통 날 목록 (상위 10개):")
        top_active = active.nlargest(10, "mean")
        for _, row in top_active.iterrows():
            print(f"    {row['date']}  mean={row['mean']:.2f}  max={row['max']:.2f}")

    # 결과 저장
    out_path = "storm_day_analysis.csv"
    df_res.sort_values("mean", ascending=False).to_csv(out_path, index=False)
    print(f"\n  전체 결과 저장: {out_path}")

    # oversampling 가중치 출력
    print(f"\n{'='*55}")
    print(f"  Oversampling 가중치 추천")
    print(f"{'='*55}")
    print(f"  조용한 날: weight = 1.0")
    if len(active) > 0:
        w_active = len(quiet) / (len(active) + 1e-6)
        print(f"  보통 날  : weight = {w_active:.1f}")
    if len(storm) > 0:
        w_storm = len(quiet) / (len(storm) + 1e-6)
        print(f"  폭풍 날  : weight = {w_storm:.1f}")
    print(f"  → 폭풍 날이 조용한 날과 동일한 빈도로 학습에 포함됨")


if __name__ == "__main__":
    main()