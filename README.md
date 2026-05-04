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

This repo does **not** include the raw Stack Exchange export CSVs (they can be large). Instead, you generate them locally via SEDE and place them under `data/`.

#### Generate raw CSVs from SEDE (Stack Exchange Data Explorer)
1) Open SEDE for Stack Overflow: https://data.stackexchange.com/stackoverflow/query/new
2) Open the query file in this repo: `queries/sede_sql_error_qa.sql`
3) Copy/paste the SQL into SEDE.
4) Click **Run Query**.
5) Click **Download as CSV** and save the file(s) into:
	- `data/sql_data/QueryResults(1).csv`
	- `data/sql_data/QueryResults(4).csv`
	- `data/sql_data/QueryResults(5).csv`

Notes:
- If you only export one file, either rename it to one of the expected filenames above or update `QUERY_RESULTS_FILES` in `src/config_sql.py`.
- You can increase `TOP 10` in the query to export more rows.

See `data/README.md` for details.

These CSVs are intentionally **not included in git** (they can be large). See `data/README.md` for details.

### 3) How to generate the raw CSVs (SEDE)
The query used to generate the raw `QueryResults*.csv` files is committed here:
- `queries/sede_sql_qa_query.sql`

Steps:
1. Open Stack Exchange Data Explorer (SEDE) for Stack Overflow: `https://data.stackexchange.com/stackoverflow/`
2. Create a new query.
3. Paste the SQL from `queries/sede_sql_qa_query.sql`.
4. (Optional) Increase `TOP 10` to a larger number to fetch more rows.
5. Run the query.
6. Export/download the results as CSV.
7. Save the downloaded CSV(s) under `data/sql_data/` using the filenames configured in `src/config_sql.py` (`QUERY_RESULTS_FILES`).

Notes:
- If your dataset is very large, you can run multiple SEDE queries (e.g., different `TOP` values or filters) and save them as separate `QueryResults(...).csv` files listed in `QUERY_RESULTS_FILES`.

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
