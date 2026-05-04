"""Encoder-Decoder model for question-answer generation."""
import torch
import torch.nn as nn

class QuestionEncoder(nn.Module):
    """LSTM Encoder for questions."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class AnswerDecoder(nn.Module):
    """LSTM Decoder for answers."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden, cell):
        embedded = self.dropout(self.embedding(x))
        out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        out = self.fc(out)
        return out, hidden, cell

def create_qa_model(question_vocab_size, answer_vocab_size, embed_size, 
                   hidden_size, num_layers, device, dropout=0.3):
    """Create and initialize question encoder and answer decoder."""
    encoder = QuestionEncoder(
        question_vocab_size, embed_size, hidden_size, num_layers, dropout
    ).to(device)
    
    decoder = AnswerDecoder(
        answer_vocab_size, embed_size, hidden_size, num_layers, dropout
    ).to(device)
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    
    print(f"\nModel Architecture:")
    print(f"  Question Encoder: {encoder_params:,} parameters")
    print(f"  Answer Decoder:   {decoder_params:,} parameters")
    print(f"  Total:            {encoder_params + decoder_params:,} parameters")
    print(f"  Dropout:          {dropout}")
    
    return encoder, decoder

if __name__ == "__main__":
    from config_sql import EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DEVICE, DROPOUT
    
    # Test model creation
    q_vocab_size = 10000
    a_vocab_size = 15000
    
    encoder, decoder = create_qa_model(
        q_vocab_size, a_vocab_size,
        EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DEVICE, DROPOUT
    )
    
    print(f"\nEncoder: {encoder}")
    print(f"\nDecoder: {decoder}")
