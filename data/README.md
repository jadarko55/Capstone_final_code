# Data directory

This folder is **ignored by git** by default (`.gitignore`), because it may contain large or private data.

## Expected raw inputs
Place your raw query export CSVs under:

- `data/sql_data/QueryResults(1).csv`
- `data/sql_data/QueryResults(4).csv`
- `data/sql_data/QueryResults(5).csv`

These filenames are configured in `src/config_sql.py` as `QUERY_RESULTS_FILES`.

## Generated outputs
After running:

```powershell
python src/preprocess_sql.py
```

You should see:
- `data/preprocessed_sql_qa.csv`
- `data/train_sql_qa.csv`
- `data/test_sql_qa.csv`
