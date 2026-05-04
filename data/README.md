# Data directory

This folder is **ignored by git** by default (`.gitignore`), because it may contain large or private data.

## How to fetch the raw data (SEDE)
The SEDE query used to generate the raw CSVs is in:
- `queries/sede_sql_qa_query.sql`

Run it on Stack Exchange Data Explorer for Stack Overflow:
`https://data.stackexchange.com/stackoverflow/`

Export/download the results as CSV and place them in `data/sql_data/` using the filenames listed below.

## Expected raw inputs
Generate your raw query export CSVs using SEDE and the query in `queries/sede_sql_error_qa.sql`, then place them under:

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
