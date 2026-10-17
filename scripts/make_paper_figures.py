#!/usr/bin/env python3
"""Redraw the paper figures from results/ (no GPU).  Usage: python scripts/make_paper_figures.py   (run from the repo root)"""
import json, glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7})
RUNS, OUT = 'results/runs/', 'results/figures_paper/'; os.makedirs(OUT, exist_ok=True)
VL = {'none': 'No balancing', 'classweight': 'Class-weighted', 'ros': 'Random oversamp.', 'smote_nn': 'Latent SMOTE'}
COL = {'none': '#4C72B0', 'classweight': '#DD8452', 'ros': '#55A868', 'smote_nn': '#C44E52'}; vs = list(VL)
R = pd.DataFrame([json.load(open(f)) for f in glob.glob(RUNS + 'main_distilbert-base-uncased_*.json')])
H = pd.read_csv('results/tables/hparam_search.csv', index_col=0)
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4)); x = np.arange(len(H))
lbl = [f"{r.lr*1e5:g}e-5\n{r.dropout}\n{int(r.batch_size)}" for r in H.itertuples()]
for ax, (m, t) in zip(axes, [('val_acc', 'Validation accuracy (%)'), ('val_f1', 'Validation macro-F1 (%)'), ('val_auc', 'Validation AUC (%)')]):
    mean, sd = H[m].values * 100, H[m + '_sd'].values * 100
    bars = ax.bar(x, mean, yerr=sd, capsize=2, color='#9db8d9', edgecolor='black', linewidth=.5)
    for b, sel in zip(bars, H['selected']):
        if sel: b.set_facecolor('#2f6db3'); b.set_hatch('////')
    ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_title(t); ax.grid(axis='y', alpha=.3)
    ax.set_ylim((mean - sd).min() - 0.8, min(100, (mean + sd).max() + 0.5))
fig.tight_layout(); fig.savefig(OUT + 'fig_hparam.pdf', bbox_inches='tight'); plt.close()
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
for v in vs:
    sub = R[R.variant == v]
    for ax, key in zip(axes, ['train_loss', 'val_auc', 'val_acc']):
        Hh = np.array([[h[key] for h in hist] for hist in sub['history']]); ep = np.arange(1, Hh.shape[1] + 1)
        ax.plot(ep, Hh.mean(0), marker='o', ms=3, color=COL[v], label=VL[v])
        ax.fill_between(ep, Hh.mean(0) - Hh.std(0), Hh.mean(0) + Hh.std(0), color=COL[v], alpha=.15)
for ax, t in zip(axes, ['Training loss', 'Validation AUC', 'Validation accuracy']):
    ax.set_title(t); ax.set_xlabel('Epoch'); ax.grid(alpha=.3); ax.set_xticks([1, 2, 3, 4])
axes[0].set_yscale('log'); axes[1].legend(loc='lower right', frameon=True)
fig.tight_layout(); fig.savefig(OUT + 'fig_curves.pdf', bbox_inches='tight'); plt.close()
z = np.load(RUNS + 'main_distilbert-base-uncased_smote_nn_42.npz'); y, p = z['labels'], z['probs']; pred = p >= .5
cm = np.array([[((y == 0) & ~pred).sum(), ((y == 0) & pred).sum()], [((y == 1) & ~pred).sum(), ((y == 1) & pred).sum()]]); pct = cm / cm.sum(1, keepdims=True) * 100
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9)); axes[0].imshow(pct, cmap='Blues', vmin=0, vmax=100)
for i in range(2):
    for j in range(2): axes[0].text(j, i, f"{cm[i,j]:,}\n({pct[i,j]:.1f}%)", ha='center', va='center', color='white' if pct[i, j] > 50 else 'black', fontsize=9)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Human', 'AI']); axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['Human', 'AI'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual'); axes[0].set_title(f'Test confusion matrix (n = {len(y):,})')
for v in vs:
    zz = np.load(RUNS + f'main_distilbert-base-uncased_{v}_42.npz'); f, t, _ = roc_curve(zz['labels'], zz['probs']); axes[1].plot(f * 100, t * 100, color=COL[v], lw=1.3, label=VL[v])
axes[1].plot([0, 100], [0, 100], 'k--', lw=.6); axes[1].set_xlim(0, 10); axes[1].set_ylim(90, 100.2)
axes[1].set_xlabel('False-positive rate (%)'); axes[1].set_ylabel('True-positive rate (%)'); axes[1].set_title('ROC, low-FPR region (seed 42)'); axes[1].legend(loc='lower right'); axes[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + 'fig_confusion_roc.pdf', bbox_inches='tight'); plt.close(); print('wrote figures to', OUT)
