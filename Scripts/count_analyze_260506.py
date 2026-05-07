"""
KSEM_L0 Raw Count CSV 기본 분석 스크립트

파일 구조:
  - 파일명: YYYYMMDD_PD{1,2,3}_Raw Count.csv
  - Time 컬럼: 1분 간격 datetime
  - 컬럼 형식: A0(0), A1(50), ..., B0(0), B1(50), ... (괄호 안 = 입자 직경 nm)
  - 채널 A, B 각각 128개 bin + Trash bin
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ─── 설정 ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


# ─── 1. 파일 로드 ────────────────────────────────────────────────────────────
def load_all_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    """모든 CSV 파일을 로드해 {파일명: DataFrame} 딕셔너리로 반환"""
    files = sorted(data_dir.glob("*_Raw Count.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")

    dfs = {}
    for f in files:
        df = pd.read_csv(f)
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time").sort_index()
        dfs[f.stem] = df
        print(f"  로드: {f.name}  →  {df.shape[0]}행 × {df.shape[1]}열")
    return dfs


def parse_channel_cols(df: pd.DataFrame, channel: str = "A") -> tuple[list[str], list[float]]:
    """채널(A 또는 B)에 해당하는 컬럼과 직경 목록 반환 (Trash·0nm 제외)"""
    pattern = re.compile(rf"^{channel}\d+\((\d+)\)$")
    cols, diams = [], []
    for col in df.columns:
        m = pattern.match(col)
        if m and int(m.group(1)) > 0:
            cols.append(col)
            diams.append(float(m.group(1)))
    # 직경 순 정렬
    paired = sorted(zip(diams, cols))
    diams_sorted = [d for d, _ in paired]
    cols_sorted = [c for _, c in paired]
    return cols_sorted, diams_sorted


# ─── 2. 기본 통계 ────────────────────────────────────────────────────────────
def print_summary(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for name, df in dfs.items():
        a_cols, a_diams = parse_channel_cols(df, "A")
        b_cols, b_diams = parse_channel_cols(df, "B")

        total_A = df[a_cols].sum(axis=1)
        total_B = df[b_cols].sum(axis=1)

        print(f"\n[{name}]")
        print(f"  Period   : {df.index[0]}  ~  {df.index[-1]}")
        print(f"  Records  : {len(df)} min")
        print(f"  Ch-A bins: {len(a_cols)}  |  Ch-B bins: {len(b_cols)}")
        print(f"  Ch-A total  mean={total_A.mean():.1f}  max={total_A.max():.1f}")
        print(f"  Ch-B total  mean={total_B.mean():.1f}  max={total_B.max():.1f}")


# ─── 3. 시계열 플롯 ──────────────────────────────────────────────────────────
def plot_timeseries(dfs: dict[str, pd.DataFrame], top_n: int = 5):
    """파일별로 채널 A의 상위 n개 bin 시계열을 저장"""
    for name, df in dfs.items():
        a_cols, a_diams = parse_channel_cols(df, "A")
        if not a_cols:
            continue

        # 평균 카운트 기준 상위 bin 선택
        means = df[a_cols].mean()
        top_cols = means.nlargest(top_n).index.tolist()

        fig, ax = plt.subplots(figsize=(12, 4))
        for col in top_cols:
            m = re.search(r"\((\d+)\)", col)
            label = f"{m.group(1)} nm" if m else col
            ax.plot(df.index, df[col], lw=0.8, label=label)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_xlabel("Time")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}  —  Channel A Top-{top_n} Bins")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        out = OUTPUT_DIR / f"{name}_timeseries.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  저장: {out.name}")


# ─── 4. 입자 크기 분포 (평균) ────────────────────────────────────────────────
def plot_size_distribution(dfs: dict[str, pd.DataFrame]):
    """모든 파일의 채널 A 평균 크기 분포를 한 그래프에 표시"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in dfs.items():
        a_cols, a_diams = parse_channel_cols(df, "A")
        if not a_cols:
            continue
        mean_counts = df[a_cols].mean().values
        ax.plot(a_diams, mean_counts, marker="o", markersize=3, lw=1, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Particle Diameter (nm)")
    ax.set_ylabel("Mean Count")
    ax.set_title("Channel A — Mean Particle Size Distribution (All Files)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    out = OUTPUT_DIR / "size_distribution_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  저장: {out.name}")


# ─── 5. 일별·기기별 총 카운트 히트맵 ────────────────────────────────────────
def plot_hourly_heatmap(dfs: dict[str, pd.DataFrame]):
    """파일별로 채널 A 총 카운트의 시간대별 분포(히트맵) 저장"""
    for name, df in dfs.items():
        a_cols, _ = parse_channel_cols(df, "A")
        if not a_cols:
            continue

        total = df[a_cols].sum(axis=1).rename("total")
        pivot = (
            total.to_frame()
            .assign(hour=total.index.hour, minute=total.index.minute)
            .pivot_table(index="minute", columns="hour", values="total", aggfunc="mean")
        )

        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       extent=[-0.5, 23.5, -0.5, 59.5], cmap="YlOrRd")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Minute")
        ax.set_title(f"{name}  —  Channel A Total Count Heatmap")
        fig.colorbar(im, ax=ax, label="Mean Count")
        fig.tight_layout()
        out = OUTPUT_DIR / f"{name}_heatmap.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  저장: {out.name}")


# ─── 6. PD 기기별 비교 (같은 날짜) ──────────────────────────────────────────
def plot_pd_comparison(dfs: dict[str, pd.DataFrame]):
    """같은 날짜의 PD1/PD2/PD3 채널 A 총 카운트를 비교"""
    # 날짜별로 그룹핑
    date_groups: dict[str, dict] = {}
    for name, df in dfs.items():
        m = re.match(r"(\d{8})_(PD\d+)", name)
        if not m:
            continue
        date, pd_id = m.groups()
        date_groups.setdefault(date, {})[pd_id] = df

    for date, pds in date_groups.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        for pd_id, df in sorted(pds.items()):
            a_cols, _ = parse_channel_cols(df, "A")
            if not a_cols:
                continue
            total = df[a_cols].sum(axis=1)
            ax.plot(df.index, total, lw=0.8, label=pd_id)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_xlabel("Time")
        ax.set_ylabel("Total Count (Channel A)")
        ax.set_title(f"{date}  —  Channel A Total Count by PD Device")
        ax.legend()
        fig.tight_layout()
        out = OUTPUT_DIR / f"{date}_PD_comparison.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  저장: {out.name}")


# ─── 메인 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading CSV files...")
    dfs = load_all_files(DATA_DIR)

    print_summary(dfs)

    print("\nGenerating time-series plots...")
    plot_timeseries(dfs, top_n=5)

    print("\nGenerating size distribution plot...")
    plot_size_distribution(dfs)

    print("\nGenerating heatmaps...")
    plot_hourly_heatmap(dfs)

    print("\nGenerating PD comparison plots...")
    plot_pd_comparison(dfs)

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")
