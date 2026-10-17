# Data

We use two public corpora. They are **not** included in this repository; download them from the original sources and respect their terms.

## 1. GRiD — GPT Reddit Dataset (the "Reddit" part)
- Paper: Z. Qazi, W. Shiao, E. E. Papalexakis, "GPT-generated text detection: Benchmark dataset and tensor-based detection method," WWW '24 Companion, doi:10.1145/3589335.3651513.
- Download: <https://github.com/madlab-ucr/GriD> (folder `reddit_datasets`; we used the **filtered** CSV, saved as `reddit_filtered_dataset.csv`).
- Content: 6,513 texts = 5,145 human comments (r/AskHistorians, r/AskScience, r/ExplainLikeImFive; dated before Nov 2022) + 1,368 GPT-3.5-turbo answers (output capped at 100 tokens).
- Columns our code expects: `Data` (text) and `Labels` (0 = human, 1 = AI). Rename the label column if your copy uses `Label`.
- Licence: the repository contains no licence file.

## 2. AIGTxt — AI-Generated Scientific Text Dataset
- Dataset: B. Alhijawi, R. Jarrar, S. Awajan, A. AbuAlRub, A. Bader, Mendeley Data, v1, doi:10.17632/y9bj7734vf.1.
- Associated article: Alhijawi et al., "Deep learning detection method for large language models-generated scientific content," Neural Computing and Applications 37(1):91–104, 2025.
- Content: paragraphs from pre-2022 academic papers in ten domains, each with a ChatGPT rewrite (and a Mixed text, which we do **not** use). 3,607 rows.
- Columns our code expects: `Human-Generated` and `ChatGPT-Generated`.
- Licence: see the "Licence" section of the Mendeley page.

## How the corpus is built (see `load_data` in notebook 01 / `scripts/verify_split.py`)
Rows are concatenated in this order: GRiD rows, AIGTxt human rows, AIGTxt ChatGPT rows. Then: drop missing texts, drop texts of ≤10 characters, drop rows whose normalized text (lower-cased, whitespace-collapsed) appears with conflicting labels, drop exact duplicates. Result: 13,690 texts (8,718 human, 4,972 AI). `splits/split_assignments.csv` is indexed by the row position after this cleaning.

## Checksums of the files we used
Run `python scripts/hash_data.py reddit_filtered_dataset.csv AIGTxt.csv` and record the output here:

- `reddit_filtered_dataset.csv`: sha256 = 80ed70e0163d365bef4172759a4712f102919badbecc3d9c1dfb86557c0f39ac
- `AIGTxt.csv`: sha256 = 62c944af54350338c6da4b8fa45f319548f04a217a123d36b17b96bb25ed3b2d
