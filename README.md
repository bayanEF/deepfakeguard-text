# DeepFakeGuard-Text

Code, split assignments and result files for the paper

> **DeepFakeGuard-Text: A Lightweight Transformer Detector for AI-Generated Text and an Evaluation of Embedding-Space SMOTE**
> B. Alfarrayeh, M. Alashqar, M. Sabeha, F. Abu Al-Soud, Q. Abu Al-Haija — Jordan University of Science and Technology (3ICT 2026).

DeepFakeGuard-Text is a binary detector of **AI-generated vs. human-written text** (authorship, *not* truthfulness): a DistilBERT encoder fine-tuned end-to-end with a two-layer head. The paper evaluates it under a leakage-controlled protocol and compares four imbalance-handling strategies (none, class-weighted loss, random oversampling, SMOTE on frozen-encoder embeddings mapped back to nearest real texts).

## Repository layout

```
notebooks/
  01_main_experiments.ipynb              data audit, group-aware split, hyper-parameter search, 4-way ablation (3 seeds),
                                         5-fold CV, cross-source tests, latency, figures, results export
  02_supplement_roberta_paraphrase.ipynb RoBERTa-base / BERT-base baselines and the paraphrase-attack pilot
splits/split_assignments.csv             row index -> source, label, group, split (train/val/test). NO text.
scripts/
  verify_split.py                        rebuilds the split from the raw CSVs and checks it against split_assignments.csv
  hash_data.py                           prints SHA-256 / row counts of the raw CSVs
  make_paper_figures.py                  redraws the paper figures from results/
results/
  RESULTS_FOR_PAPER.txt                  every table printed by notebook 01
  RESULTS_SUPPLEMENT.txt                 every table printed by notebook 02
  tables/*.csv                           summary tables (incl. summary_main_all_encoders.csv, latency, hyper-parameter search, CV, cross-source)
  tables/paraphrase_pilot_scores.csv     detector scores before/after paraphrasing (no text)
  runs/*.json                            per-run metrics + per-epoch history (one file per run)
  runs/*.npz                             per-run test probabilities and labels (no text)
  figures/                               figures as exported by the notebook
  figures_paper/                         the figures used in the paper (redrawn by scripts/make_paper_figures.py)
DATA.md                                  where to get the two datasets and what format the code expects
```

## Headline results (test split, 3 seeds, threshold 0.5)

| Model / strategy | Accuracy (%) | AUC (%) |
|---|---|---|
| DistilBERT, none | 96.59 ± 1.26 | 99.62 ± 0.02 |
| DistilBERT, class-weighted | 96.64 ± 0.90 | 99.63 ± 0.02 |
| DistilBERT, random oversampling | 97.09 ± 1.09 | 99.67 ± 0.05 |
| DistilBERT, latent SMOTE | 97.43 ± 0.21 | 99.65 ± 0.03 |
| BERT-base, none | 97.13 ± 0.50 | 99.70 ± 0.02 |
| RoBERTa-base, none | 97.28 ± 1.00 | 99.80 ± 0.01 |

The effect of latent SMOTE is small and not consistent across seeds (and absent in 5-fold CV: 96.59 vs 96.54). Cross-source transfer is weak (AUC 87.6 for AIGTxt→GRiD; 76.5–81.3 for GRiD→AIGTxt). See the paper and `results/` for all numbers, caveats and limitations.

## Data (not included)

The two datasets are **not redistributed** here. See [`DATA.md`](DATA.md) for sources and expected files. After downloading them, check that you have the same data as the paper:

```bash
pip install pandas scikit-learn
python3 scripts/verify_split.py --reddit reddit_filtered_dataset.csv --aigtxt AIGTxt.csv --splits splits/split_assignments.csv
# expected: 13,729 raw rows -> 13,690 texts (8,718 human, 4,972 AI); ends with "PASS: your data matches the released split file."
```

`splits/split_assignments.csv` is the **source of truth** for the split (train 9,581 | val 1,372 | test 2,737). The script also re-derives a split with your scikit-learn version for information only: `StratifiedGroupKFold` can produce slightly different folds in different scikit-learn versions (a re-derivation on a different version gave 9,582 / 1,370 / 2,738), so for exact reproduction always use the released file rather than recomputing.

## Reproducing the experiments

1. Open `notebooks/01_main_experiments.ipynb` in Google Colab with a GPU (we used a Tesla T4). Upload the two CSVs and `splits/split_assignments.csv` to `/content/` (the notebooks then use the released split; optionally mount Google Drive and point `CFG['save_dir']` to it so finished runs survive a disconnect).
2. Run the cells top to bottom. Finished runs are cached and skipped on re-run.
3. For the baselines and paraphrase pilot, run `notebooks/02_supplement_roberta_paraphrase.ipynb` (upload `deepfakeguard_results_pack.zip` produced by notebook 01 when asked).
4. `python scripts/make_paper_figures.py` redraws the paper figures from `results/`.

Fixed settings: seeds 42/43/44; AdamW (lr 3e-5, weight decay 0.01), batch size 16, dropout 0.3/0.15, max length 256, 4 epochs, linear schedule with 10% warm-up, gradient clipping 1.0, FP16 autocast, checkpoint chosen by **validation** AUC, test split scored once. Split: 70/10/20, stratified and **group-aware** (a human paragraph and its ChatGPT rewrite in AIGTxt share one group).

**Environment.** Google Colab, Tesla T4, PyTorch 2.11.0+cu130. Other package versions were not recorded; `requirements.txt` lists the packages without pins. GPU training is not bit-exact, so re-runs can differ slightly from the stored results.

## Not included

Model checkpoints (large; the headline model can be retrained in ~6 minutes), the raw datasets and any API keys. The paraphrase pilot used `Qwen/Qwen2.5-3B-Instruct` (research licence), downloaded at run time.

## Licences

Code: MIT (see `LICENSE`). Datasets: governed by their own terms (see `DATA.md`). The Qwen model used in the pilot is under its own licence.

## Citation

See `CITATION.cff`. Please also cite the original datasets (GRiD; AIGTxt) listed in `DATA.md`.
