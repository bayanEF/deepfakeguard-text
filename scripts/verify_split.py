#!/usr/bin/env python3
"""Check that your copies of the two raw CSVs are the ones used in the paper, using splits/split_assignments.csv as the source of truth.
No GPU or torch needed.  Usage:
    python3 scripts/verify_split.py --reddit reddit_filtered_dataset.csv --aigtxt AIGTxt.csv --splits splits/split_assignments.csv

PASS (exit 0)  = same rows, labels, sources and groups as the released split file, and no group straddles splits.
The script also re-derives the split with YOUR scikit-learn version for information only: StratifiedGroupKFold can
give slightly different folds in different scikit-learn versions, so use splits/split_assignments.csv (not a re-derivation)
for exact reproduction.
Expected (paper): 13,729 raw rows -> 13,690 texts; train 9,581 / val 1,372 / test 2,737.
"""
import argparse, sys, numpy as np, pandas as pd, sklearn
from sklearn.model_selection import StratifiedGroupKFold

def load_data(cfg):
    rd = pd.read_csv(cfg['reddit_csv'])[['Data', 'Labels']].rename(columns={'Data': 'text', 'Labels': 'label'})
    rd['source'] = 'reddit'
    rd['group'] = ['reddit_%d' % i for i in range(len(rd))]
    aig = pd.read_csv(cfg['aig_csv'])
    # A human paragraph and its ChatGPT rewrite share one AIGTxt row -> same group
    h = pd.DataFrame({'text': aig['Human-Generated'], 'label': 0, 'source': 'aigtxt'})
    h['group'] = ['aig_%d' % i for i in aig.index]
    a = pd.DataFrame({'text': aig['ChatGPT-Generated'], 'label': 1, 'source': 'aigtxt'})
    a['group'] = ['aig_%d' % i for i in aig.index]
    df = pd.concat([rd, h, a], ignore_index=True)

    log = []
    def snap(step, d):
        log.append(dict(step=step, total=len(d), human=int((d.label == 0).sum()), ai=int((d.label == 1).sum())))
    snap('raw rows', df)
    df = df.dropna(subset=['text']).copy()
    df['text'] = df['text'].astype(str).str.strip()
    snap('after dropna', df)
    df = df[df['text'].str.len() > 10].copy()
    snap('after length > 10 chars', df)
    df['norm'] = df['text'].str.lower().str.replace(r'\s+', ' ', regex=True)
    n_lab = df.groupby('norm')['label'].nunique()
    conflict = set(n_lab[n_lab > 1].index)
    df = df[~df['norm'].isin(conflict)]
    snap('after dropping label-conflicting duplicates', df)
    df = df.drop_duplicates('norm').reset_index(drop=True)
    snap('after exact-duplicate removal', df)
    return df, pd.DataFrame(log)

def make_splits(df, seed=42, test_folds=5, val_folds=8):
    """Group-aware, stratified train/val/test. Texts that share a group (same AIGTxt row) never straddle splits."""
    y, g = df['label'].values, df['group'].values
    idx = np.arange(len(df))
    sg = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    dev, test = next(iter(sg.split(idx, y, g)))
    sg2 = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(iter(sg2.split(dev, y[dev], g[dev])))
    tr, va = dev[tr_rel], dev[va_rel]
    assert not (set(g[tr]) & set(g[va])) and not (set(g[tr]) & set(g[test])) and not (set(g[va]) & set(g[test]))
    return tr, va, test


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reddit', required=True); ap.add_argument('--aigtxt', required=True)
    ap.add_argument('--splits', default='splits/split_assignments.csv'); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    df, log = load_data(dict(reddit_csv=a.reddit, aig_csv=a.aigtxt))
    print(log.to_string(index=False)); print()
    A = pd.read_csv(a.splits, index_col=0)
    same_len = len(A) == len(df)
    same_rows = same_len and bool((A['label'].values == df['label'].values).all() and (A['source'].values == df['source'].values).all()
                                  and (A['group'].values == df['group'].values).all())
    straddle = int(A.groupby('group')['split'].nunique().gt(1).sum())
    print(f"rows: split file {len(A)} vs your data {len(df)} -> {'OK' if same_len else 'DIFFERENT'}")
    print(f"labels / sources / groups identical row by row -> {'OK' if same_rows else 'DIFFERENT'}")
    print(f"groups that straddle splits in the split file -> {straddle} ({'OK' if straddle == 0 else 'PROBLEM'})")
    for nm in ('train', 'val', 'test'):
        m = A['split'] == nm
        print(f"  {nm:5s}: {int(m.sum()):6,d} texts, AI share {A.loc[m, 'label'].mean()*100:.1f}%")
    tr, va, te = make_splits(df, seed=a.seed)
    same_split = (set(np.where(A['split'] == 'train')[0]) == set(tr) and set(np.where(A['split'] == 'val')[0]) == set(va)
                  and set(np.where(A['split'] == 'test')[0]) == set(te))
    print(f"\n[info] re-derived with scikit-learn {sklearn.__version__}: train {len(tr)} | val {len(va)} | test {len(te)} -> "
          f"{'identical to the split file' if same_split else 'differs from the split file (normal across scikit-learn versions; use the split file)'}")
    ok = same_len and same_rows and straddle == 0
    print('\nPASS: your data matches the released split file.' if ok else '\nFAIL: your CSVs differ from the ones used in the paper.')
    sys.exit(0 if ok else 1)
