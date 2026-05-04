"""Configuration settings for the SQL question-answer model."""
from pathlib import Path
import torch

# Project paths (resolve relative to repo root, not CWD)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

# Model hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 20

# Paths
SQL_DATA_DIR = str(DATA_DIR / "sql_data")
QUERY_RESULTS_FILES = [
    'QueryResults(1).csv',
    'QueryResults(4).csv', 
    'QueryResults(5).csv'
]
PREPROCESSED_SQL_DATA_PATH = str(DATA_DIR / "preprocessed_sql_qa.csv")
TRAIN_SQL_DATA_PATH = str(DATA_DIR / "train_sql_qa.csv")
TEST_SQL_DATA_PATH = str(DATA_DIR / "test_sql_qa.csv")
SQL_CHECKPOINT_PATH = str(MODELS_DIR / "sql_qa_checkpoint.pth")
SQL_MODEL_PATH = str(MODELS_DIR / "sql_qa_model.pth")
SQL_MODELS_DIR = str(MODELS_DIR)

# Training parameters
MAX_QUESTION_LENGTH = 100
MAX_ANSWER_LENGTH = 500
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Text preprocessing
MIN_QUESTION_LENGTH = 5
MIN_ANSWER_LENGTH = 10
MAX_VOCAB_SIZE = 35000

# Gradient clipping
GRAD_CLIP_MAX_NORM = 5.0
