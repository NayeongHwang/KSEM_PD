# diagnose_only.py
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR  = Path(__file__).parent
ALL_DATES = ["20240510","20240511","20240512","20240513"]
TRASH     = ["A127(Trash)", "B127(Trash)"]

frames = []
for date in ALL_DATES:
    for pd_id in [1,2,3]:
        f = DATA_DIR / f"{date}_PD{pd_id}_Raw Count.csv"
        if f.exists():
            df = pd.read_csv(f)
            frames.append(df)

df_all    = pd.concat(frames, ignore_index=True)
spec_cols = [c for c in df_all.columns if c not in ["Time"]+TRASH]
X         = np.log1p(df_all[spec_cols].values.astype(np.float32))

means = X.mean(axis=0)
top5  = np.argsort(means)[-5:][::-1]
print("log1p 후 평균값 TOP5 채널:")
for i in top5:
    print(f"  {spec_cols[i]:20s}  mean={means[i]:.3f}  std={X[:,i].std():.3f}")
print(f"\n전체 mean={means.mean():.3f}, max={X.max():.3f}")