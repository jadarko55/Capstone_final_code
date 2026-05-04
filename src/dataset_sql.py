"""PyTorch Dataset for SQL question-answer pairs."""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from vocabulary_sql import tokenize_text
from config_sql import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

class QADataset(Dataset):
    """Dataset class for question-answer pairs."""
    
    def __init__(self, questions, answers, question_word2int, answer_word2int):
        self.questions = questions
        self.answers = answers
        self.question_word2int = question_word2int
        self.answer_word2int = answer_word2int
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Tokenize and encode question
        question_tokens = tokenize_text(question)
        question_tensor = torch.tensor(
            [self.question_word2int.get(word, self.question_word2int[UNK_TOKEN]) 
             for word in question_tokens]
            + [self.question_word2int[EOS_TOKEN]],
            dtype=torch.long
        )
        
        # Tokenize and encode answer
        answer_tokens = tokenize_text(answer)
        answer_tensor = torch.tensor(
            [self.answer_word2int.get(word, self.answer_word2int[UNK_TOKEN]) 
             for word in answer_tokens]
            + [self.answer_word2int[EOS_TOKEN]],
            dtype=torch.long
        )
        
        return question_tensor, answer_tensor

def collate_fn_qa(batch, question_pad_idx, answer_pad_idx):
    """Custom collate function for padding sequences."""
    question_batch, answer_batch = zip(*batch)
    
    question_padded = pad_sequence(
        question_batch,
        batch_first=True,
        padding_value=question_pad_idx
    )
    answer_padded = pad_sequence(
        answer_batch,
        batch_first=True,
        padding_value=answer_pad_idx
    )
    
    return question_padded, answer_padded

def create_qa_dataloader(questions, answers, question_word2int, answer_word2int, 
                        batch_size, shuffle=True):
    """Create DataLoader for Q&A pairs."""
    dataset = QADataset(questions, answers, question_word2int, answer_word2int)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn_qa(
            batch,
            question_word2int[PAD_TOKEN],
            answer_word2int[PAD_TOKEN]
        )
    )
    
    print(f"Created DataLoader: {len(dataset)} samples, {len(dataloader)} batches")
    return dataloader

if __name__ == "__main__":
    import pandas as pd
    from vocabulary_sql import create_qa_vocabularies
    from config_sql import TRAIN_SQL_DATA_PATH, BATCH_SIZE
    
    # Test dataset creation
    train_df = pd.read_csv(TRAIN_SQL_DATA_PATH)
    questions = train_df['cleaned_question'].tolist()[:100]
    answers = train_df['cleaned_answer'].tolist()[:100]
    
    (q_word2int, q_int2word, 
     a_word2int, a_int2word) = create_qa_vocabularies(questions, answers)
    
    dataloader = create_qa_dataloader(
        questions, answers, q_word2int, a_word2int, BATCH_SIZE
    )
    
    # Test batch
    for q_batch, a_batch in dataloader:
        print(f"Question batch shape: {q_batch.shape}")
        print(f"Answer batch shape: {a_batch.shape}")
        break
