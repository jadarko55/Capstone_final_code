"""Data preprocessing for SQL question-answer pairs."""
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from config_sql import (
    SQL_DATA_DIR, QUERY_RESULTS_FILES, TRAIN_TEST_SPLIT, 
    RANDOM_SEED, MIN_QUESTION_LENGTH, MIN_ANSWER_LENGTH,
    PREPROCESSED_SQL_DATA_PATH, TRAIN_SQL_DATA_PATH, TEST_SQL_DATA_PATH
)

def load_sql_query_results(data_dir):
    """Load and aggregate all QueryResults CSV files."""
    all_data = []
    
    for file in QUERY_RESULTS_FILES:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"Loading {file}...")
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"  Loaded {len(df)} rows")
        else:
            print(f"Warning: {file} not found at {file_path}")
    
    if not all_data:
        raise FileNotFoundError(f"No QueryResults files found in {data_dir}")
    
    # Concatenate all dataframes
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows after aggregation: {len(df_combined)}")
    print(f"Columns: {df_combined.columns.tolist()}")
    
    return df_combined

def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def filter_valid_qa_pairs(df):
    """Filter valid question-answer pairs."""
    print("\nFiltering valid Q&A pairs...")
    
    # Identify question and answer columns
    question_col = None
    answer_col = None
    
    # Print available columns for debugging
    print(f"Available columns: {df.columns.tolist()}")
    
    # Try to find question and answer columns
    for col in df.columns:
        col_lower = col.lower()
        if 'question' in col_lower and 'body' in col_lower and question_col is None:
            question_col = col
        elif 'answer' in col_lower and 'body' in col_lower and answer_col is None:
            answer_col = col
    
    # If specific columns not found, try generic body columns
    if not question_col or not answer_col:
        body_cols = [c for c in df.columns if 'body' in c.lower()]
        if len(body_cols) >= 2:
            question_col = body_cols[0]
            answer_col = body_cols[1]
            print(f"Using generic body columns: '{question_col}' and '{answer_col}'")
    
    if not question_col or not answer_col:
        raise ValueError(f"Could not identify question/answer columns. Available: {df.columns.tolist()}")
    
    print(f"Question column: '{question_col}'")
    print(f"Answer column: '{answer_col}'")
    
    # Clean text
    df['cleaned_question'] = df[question_col].apply(clean_text)
    df['cleaned_answer'] = df[answer_col].apply(clean_text)
    
    # Remove empty entries
    df = df[
        (df['cleaned_question'].str.len() >= MIN_QUESTION_LENGTH) &
        (df['cleaned_answer'].str.len() >= MIN_ANSWER_LENGTH)
    ]
    
    print(f"\nValid Q&A pairs after filtering: {len(df)}")
    print(f"Average question length: {df['cleaned_question'].str.len().mean():.1f} chars")
    print(f"Average answer length: {df['cleaned_answer'].str.len().mean():.1f} chars")
    
    return df[['cleaned_question', 'cleaned_answer']]

def preprocess_sql_qa_pipeline(data_dir=None, output_dir=None):
    """Complete preprocessing pipeline for SQL Q&A data."""
    print("="*80)
    print("SQL QUESTION-ANSWER PREPROCESSING PIPELINE")
    print("="*80)
    
    if data_dir is None:
        data_dir = SQL_DATA_DIR
    if output_dir is None:
        output_dir = os.path.dirname(PREPROCESSED_SQL_DATA_PATH)
    
    # Load and aggregate data
    df = load_sql_query_results(data_dir)
    
    # Filter and clean Q&A pairs
    df_clean = filter_valid_qa_pairs(df)
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    original_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['cleaned_question', 'cleaned_answer'])
    print(f"Removed {original_len - len(df_clean)} duplicates")
    
    # Split into train/test
    print(f"\nSplitting data (train: {(1-TRAIN_TEST_SPLIT)*100:.0f}%, test: {TRAIN_TEST_SPLIT*100:.0f}%)...")
    train_df, test_df = train_test_split(
        df_clean,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"  Train: {len(train_df)} pairs")
    print(f"  Test:  {len(test_df)} pairs")
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(TRAIN_SQL_DATA_PATH, index=False)
    test_df.to_csv(TEST_SQL_DATA_PATH, index=False)
    df_clean.to_csv(PREPROCESSED_SQL_DATA_PATH, index=False)
    
    print(f"\n✅ Data saved:")
    print(f"  Train: {TRAIN_SQL_DATA_PATH}")
    print(f"  Test:  {TEST_SQL_DATA_PATH}")
    print(f"  All:   {PREPROCESSED_SQL_DATA_PATH}")
    
    # Sample preview
    print(f"\n📋 Sample Q&A pairs:")
    for i, row in train_df.head(3).iterrows():
        print(f"\nQ: {row['cleaned_question'][:100]}...")
        print(f"A: {row['cleaned_answer'][:100]}...")
    
    return train_df, test_df, df_clean

if __name__ == "__main__":
    preprocess_sql_qa_pipeline()
