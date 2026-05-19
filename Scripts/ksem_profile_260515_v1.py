"""
KSEM PD L0.5 Profile Visualization  —  ksem_profile_v1.py
----------------------------------------------------------
폴더 구조:
  KSEM_L0_PD1/
  ├── 20240510_PD1_Raw Count.csv
  ├── 20240510_PD2_Raw Count.csv
  ├── ...
  └── output/
        └── profile_YYYYMMDD_PDx_fig*.png
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # 스크립트 위치 = workspace 루트
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 분석할 날짜 & PD 선택 ────────────────────────────────────────────────────
TARGET_DATE = "20240510"   # YYYYMMDD
TARGET_PD   = "PD1"        # PD1 / PD2 / PD3  (None 이면 전체)

# ── 채널 설정 ─────────────────────────────────────────────────────────────────
EXCLUDE_IDX = {42, 43, 44}   # coincidence logic 미정의 채널
MAX_CH_IDX  = 56             # A0~A56, B0~B56

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_csv(base_dir, date, pd_name):
    """날짜 + PD 이름으로 CSV 파일 경로 반환"""
    pattern = os.path.join(base_dir, f"{date}_{pd_name}_Raw Count.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"파일 없음: {pattern}")
    return matches[0]


def get_valid_cols(df, prefix):
    """A 또는 B 텔레스코프의 유효 채널 컬럼 리스트 반환"""
    cols = []
    for c in df.columns:
        if not c.startswith(prefix):
            continue
        try:
            idx = int(c[len(prefix):].split('(')[0])
        except ValueError:
            continue
        if 0 <= idx <= MAX_CH_IDX and idx not in EXCLUDE_IDX:
            cols.append(c)
    return cols


def load_and_preprocess(csv_path):
    """CSV 로드 → 유효 108채널 선별 → log1p 변환"""
    df = pd.read_csv(csv_path, parse_dates=['Time']).set_index('Time')
    A_cols    = get_valid_cols(df, 'A')
    B_cols    = get_valid_cols(df, 'B')
    valid_cols = A_cols + B_cols
    df_valid  = df[valid_cols].clip(lower=0)
    df_log    = np.log1p(df_valid)
    return df_valid, df_log, A_cols, B_cols


def save(fig, out_dir, tag, fignum):
    fname = os.path.join(out_dir, f"profile_{tag}_fig{fignum}.png")
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(fname)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시각화 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_channel_structure(df_log, A_cols, B_cols, tag, out_dir):
    """Fig 1. 채널 구조 — 에너지 그룹별 mean log1p CPS"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    fig.suptitle(f"[{tag}]  Channel Structure  (Valid 108 ch, A & B Telescopes)",
                 fontsize=13, fontweight='bold', y=1.01)

    groups = [
        (0,  23,  'F  (100–6000 keV e⁻)',  '#e67e22'),
        (24, 40,  'FT  (4–12 MeV)',          '#27ae60'),
        (44, 55,  'FTUO',                    '#8e44ad'),
        (56, 79,  'O-side low',              '#16a085'),  # 실제 col idx는 재매핑됨
        (80, 103, 'O-side mid',              '#2c3e50'),
        (104,116, 'OU',                      '#d35400'),
        (117,127, 'OUT',                     '#7f8c8d'),
    ]

    for ax, cols, tel in zip(axes, [A_cols, B_cols], ['A Telescope', 'B Telescope']):
        mean_v = df_log[cols].mean().values
        ax.bar(range(len(cols)), mean_v, color='#2980b9', edgecolor='none', width=0.85)
        ymax = mean_v.max() * 1.15 if mean_v.max() > 0 else 1
        for s, e, lbl, col in groups:
            e = min(e, len(cols)-1)
            if s >= len(cols): continue
            ax.axvspan(s-0.5, e+0.5, alpha=0.09, color=col)
            ax.text((s+e)/2, ymax*0.88, lbl, ha='center',
                    fontsize=7, color=col, fontweight='bold')
        ax.set_title(tel, fontweight='bold')
        ax.set_xlabel("Channel index (0–53)")
        ax.set_ylabel("Mean log1p(CPS)")
        ax.set_xlim(-1, len(cols))
        ax.set_ylim(0, ymax)

    plt.tight_layout()
    save(fig, out_dir, tag, 1)


def plot_distribution(df_valid, df_log, tag, out_dir):
    """Fig 2. Raw vs log1p 분포 비교"""
    raw_flat = df_valid.values.flatten()
    log_flat = df_log.values.flatten()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"[{tag}]  Raw CPS vs log1p — Distribution",
                 fontsize=13, fontweight='bold')

    # 2-1. Raw 히스토그램
    ax = axes[0]
    nz = raw_flat[raw_flat > 0]
    ax.hist(nz, bins=80, color='#e74c3c', alpha=0.85, edgecolor='none')
    ax.set_yscale('log')
    ax.set_xlabel("Raw CPS")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Raw CPS (non-zero values)")
    ax.text(0.97, 0.95,
            f"max = {raw_flat.max():.0f}\nzero = {100*(raw_flat==0).mean():.1f}%",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # 2-2. log1p 히스토그램
    ax = axes[1]
    ax.hist(log_flat, bins=80, color='#2980b9', alpha=0.85, edgecolor='none')
    ax.set_xlabel("log1p(CPS)")
    ax.set_ylabel("Count")
    ax.set_title("After log1p transform")
    ax.text(0.97, 0.95,
            f"range = [{log_flat.min():.2f},  {log_flat.max():.2f}]",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # 2-3. CDF 비교
    ax = axes[2]
    for data, label, color in [
        (raw_flat / (raw_flat.max() + 1e-9), 'Raw (norm)', '#e74c3c'),
        (log_flat / (log_flat.max() + 1e-9), 'log1p (norm)', '#2980b9'),
    ]:
        s = np.sort(data)
        ax.plot(s, np.linspace(0, 1, len(s)), color=color, label=label, lw=1.8)
    ax.set_xlabel("Normalized value")
    ax.set_ylabel("CDF")
    ax.set_title("CDF Comparison")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, out_dir, tag, 2)


def plot_heatmap(df_log, A_cols, B_cols, times, tag, out_dir):
    """Fig 3. 시계열 Heatmap (A + B)"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"[{tag}]  Time-series Heatmap  (log1p CPS,  1-min resolution)",
                 fontsize=13, fontweight='bold')

    n_times = len(times)
    tick_pos  = np.arange(0, n_times, 120)
    tick_labs = [times[i].strftime('%H:%M') for i in tick_pos]

    for ax, cols, tel in zip(axes, [A_cols, B_cols], ['A Telescope (54 ch)', 'B Telescope (54 ch)']):
        mat  = df_log[cols].values.T          # (54, T)
        vmax = np.percentile(mat[mat > 0], 99) if (mat > 0).any() else 1
        im = ax.imshow(mat, aspect='auto', origin='lower',
                       cmap='inferno', vmin=0, vmax=vmax,
                       extent=[0, n_times, 0, len(cols)])
        ax.set_title(tel, fontweight='bold')
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Channel index")
        plt.colorbar(im, ax=ax, label='log1p(CPS)', fraction=0.02, pad=0.01)

    plt.tight_layout()
    save(fig, out_dir, tag, 3)


def plot_timeseries(df_log, A_cols, B_cols, times, tag, out_dir):
    """Fig 4. 대표 채널 time series (에너지대별)"""
    # 컬럼명 → dict로 lookup
    col_map = {c: c for c in df_log.columns}

    # A / B 각 3개씩 대표 채널 (인덱스 기준)
    rep_targets = [
        ('A', 5,  '~119 keV  (F-low)'),
        ('A', 12, '~548 keV  (F-mid)'),
        ('A', 17, '~1626 keV (FT)'),
        ('B', 5,  '~119 keV  (F-low)'),
        ('B', 12, '~548 keV  (F-mid)'),
        ('B', 17, '~1626 keV (FT)'),
    ]
    colors_rep = ['#e74c3c','#e67e22','#f1c40f','#3498db','#2980b9','#9b59b6']

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(f"[{tag}]  Representative Channel Time Series  (log1p CPS)",
                 fontsize=13, fontweight='bold')

    n_times   = len(times)
    tick_pos  = np.arange(0, n_times, 120)
    tick_labs = [times[i].strftime('%H:%M') for i in tick_pos]

    for ax, (prefix, ch_idx, energy_lbl), color in zip(axes.flatten(), rep_targets, colors_rep):
        # 해당 인덱스의 채널 컬럼 찾기
        cols_src = A_cols if prefix == 'A' else B_cols
        target   = next((c for c in cols_src
                         if int(c[1:].split('(')[0]) == ch_idx), None)
        if target is None:
            ax.set_visible(False)
            continue

        y     = df_log[target].values
        label = f"{prefix}{ch_idx}  {energy_lbl}"
        ax.fill_between(range(n_times), y, alpha=0.25, color=color)
        ax.plot(range(n_times), y, color=color, lw=1.3, label=label)
        ax.set_title(label, fontweight='bold', color=color)
        ax.set_ylabel("log1p(CPS)")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, n_times - 1)

    for ax in axes[-1]:
        ax.set_xlabel("Time (UTC)")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, rotation=30)

    plt.tight_layout()
    save(fig, out_dir, tag, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(date, pd_name):
    print(f"\n▶ {date}  {pd_name}")
    csv_path           = find_csv(BASE_DIR, date, pd_name)
    df_valid, df_log, A_cols, B_cols = load_and_preprocess(csv_path)
    times              = df_valid.index
    tag                = f"{date}_{pd_name}"

    print(f"  유효 채널 {len(A_cols)+len(B_cols)}개 (A:{len(A_cols)}, B:{len(B_cols)})  |  "
          f"T={len(times)}  |  raw max={df_valid.values.max():.1f}")

    plot_channel_structure(df_log, A_cols, B_cols, tag, OUTPUT_DIR)
    plot_distribution(df_valid, df_log, tag, OUTPUT_DIR)
    plot_heatmap(df_log, A_cols, B_cols, times, tag, OUTPUT_DIR)
    plot_timeseries(df_log, A_cols, B_cols, times, tag, OUTPUT_DIR)


if __name__ == "__main__":
    if TARGET_PD is None:
        for pd_name in ["PD1", "PD2", "PD3"]:
            try:
                run(TARGET_DATE, pd_name)
            except FileNotFoundError as e:
                print(f"  건너뜀: {e}")
    else:
        run(TARGET_DATE, TARGET_PD)

    print("\n✅ 완료  →  output/ 폴더 확인")
