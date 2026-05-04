# Capstone Final Code — SQL Q&A (Seq2Seq)

PyTorch encoder–decoder model that learns to answer SQL-related questions from Q&A pairs.

## Project layout
- `src/` — training, preprocessing, inference code
- `notebooks/` — experimentation / training notebook
- `data/` — (ignored) raw + processed datasets
- `models/` — (ignored) saved checkpoints and final model

## Setup
### 1) Create a Python environment
PowerShell (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Data placement
This repo expects raw CSV exports in:
- `data/sql_data/QueryResults(1).csv`
- `data/sql_data/QueryResults(4).csv`
- `data/sql_data/QueryResults(5).csv`

See `data/README.md` for details.

## Preprocess
This will load the raw `QueryResults*.csv` files, clean them, deduplicate, and write:
- `data/train_sql_qa.csv`
- `data/test_sql_qa.csv`
- `data/preprocessed_sql_qa.csv`

Run from the repo root:
```powershell
python src/preprocess_sql.py
```

## Train
Training will auto-run preprocessing if `data/train_sql_qa.csv` is missing.

Run from the repo root:
```powershell
python src/train_sql.py
```

Outputs (by default):
- Checkpoint: `models/sql_qa_checkpoint.pth`
- Final model: `models/sql_qa_model.pth`

## Inference (quick test)
Runs a small qualitative test on the test split:
```powershell
python src/inference_sql.py
```

## Large files (GitHub note)
The `.pth` files in `models/` are large (hundreds of MB) and will not push to GitHub with normal git.

Options:
1) **Recommended for GitHub pushes**: keep them untracked (default). This repo’s `.gitignore` ignores `models/*.pth`.
2) **If you want to version weights**: use Git LFS and track `*.pth`.

See `models/README.md` for the exact commands.

## License
MIT (see `LICENSE`).
