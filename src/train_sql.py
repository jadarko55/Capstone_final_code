"""Training script for SQL question-answer model."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model_sql import create_qa_model
from dataset_sql import create_qa_dataloader
from vocabulary_sql import create_qa_vocabularies
from config_sql import (
    DEVICE, PAD_TOKEN, SOS_TOKEN, EMBED_SIZE, HIDDEN_SIZE,
    NUM_LAYERS, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    SQL_CHECKPOINT_PATH, SQL_MODEL_PATH, SQL_MODELS_DIR,
    DROPOUT, GRAD_CLIP_MAX_NORM, MAX_VOCAB_SIZE,
    TRAIN_SQL_DATA_PATH, TEST_SQL_DATA_PATH
)

def train_qa_model(encoder, decoder, dataloader, question_word2int, answer_word2int,
                  num_epochs=NUM_EPOCHS, start_epoch=0, checkpoint_path=SQL_CHECKPOINT_PATH):
    """Train the question-answer encoder-decoder model."""
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=answer_word2int[PAD_TOKEN])
    
    # Optimizers
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler - reduce LR when loss plateaus
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Create models directory
    os.makedirs(SQL_MODELS_DIR, exist_ok=True)
    
    # Load checkpoint if exists and compatible
    if os.path.exists(checkpoint_path) and start_epoch == 0:
        print(f"\nFound checkpoint at {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            # Check if vocabulary sizes match
            checkpoint_q_vocab = len(checkpoint.get('question_word2int', {}))
            checkpoint_a_vocab = len(checkpoint.get('answer_word2int', {}))
            current_q_vocab = len(question_word2int)
            current_a_vocab = len(answer_word2int)
            
            if checkpoint_q_vocab == current_q_vocab and checkpoint_a_vocab == current_a_vocab:
                print(f"Loading checkpoint (vocab sizes match: Q={current_q_vocab}, A={current_a_vocab})")
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
                decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"✅ Resumed from epoch {start_epoch}")
            else:
                print(f"⚠️  Checkpoint vocab mismatch!")
                print(f"   Checkpoint: Q={checkpoint_q_vocab}, A={checkpoint_a_vocab}")
                print(f"   Current:    Q={current_q_vocab}, A={current_a_vocab}")
                print(f"   Starting fresh training (checkpoint incompatible)")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")
            print(f"   Starting fresh training")
    
    # Training loop
    encoder.train()
    decoder.train()
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        batch_count = 0
        
        for i, (question_tensor, answer_tensor) in enumerate(dataloader):
            question_tensor = question_tensor.to(DEVICE)
            answer_tensor = answer_tensor.to(DEVICE)
            
            # Zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            batch_size = question_tensor.size(0)
            target_length = answer_tensor.size(1)
            
            # Encoder
            _, encoder_hidden, encoder_cell = encoder(question_tensor)
            
            # Decoder with teacher forcing
            decoder_input = torch.full(
                (batch_size, 1),
                answer_word2int[SOS_TOKEN],
                dtype=torch.long
            ).to(DEVICE)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell
            
            loss = 0
            
            for di in range(target_length):
                logits, decoder_hidden, decoder_cell = decoder(
                    decoder_input, decoder_hidden, decoder_cell
                )
                # logits shape: (batch_size, 1, vocab_size)
                logits = logits.squeeze(1)  # (batch_size, vocab_size)
                loss += loss_fn(logits, answer_tensor[:, di])
                decoder_input = answer_tensor[:, di].reshape(batch_size, 1)
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping - IMPORTANT for stability
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if (i + 1) % 10 == 0:
                avg_batch_loss = loss.item() / target_length
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {avg_batch_loss:.4f}")
        
        avg_loss = total_loss / batch_count
        
        # Update learning rate based on loss
        encoder_scheduler.step(avg_loss)
        decoder_scheduler.step(avg_loss)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs} completed - Average Loss: {avg_loss:.4f}")
        print(f"Current LR: {encoder_optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}\n")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'question_word2int': question_word2int,
                'question_int2word': {v: k for k, v in question_word2int.items()},
                'answer_word2int': answer_word2int,
                'answer_int2word': {v: k for k, v in answer_word2int.items()},
            }, checkpoint_path)
            print(f"✅ Checkpoint saved at epoch {epoch+1}\n")
    
    # Save final model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'question_word2int': question_word2int,
        'question_int2word': {v: k for k, v in question_word2int.items()},
        'answer_word2int': answer_word2int,
        'answer_int2word': {v: k for k, v in answer_word2int.items()},
        'config': {
            'embed_size': EMBED_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'question_vocab_size': len(question_word2int),
            'answer_vocab_size': len(answer_word2int)
        }
    }, SQL_MODEL_PATH)
    print(f"✅ Final model saved: {SQL_MODEL_PATH}")
    
    return encoder, decoder

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("="*80)
    print("SQL QUESTION-ANSWER MODEL TRAINING")
    print(f"Vocabulary Size: {MAX_VOCAB_SIZE}")
    print(f"Embed Size: {EMBED_SIZE}, Hidden Size: {HIDDEN_SIZE}")
    print(f"Num Layers: {NUM_LAYERS}, Dropout: {DROPOUT}")
    print(f"Learning Rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}, Grad Clip: {GRAD_CLIP_MAX_NORM}")
    print("="*80)
    
    # Load data
    train_path = TRAIN_SQL_DATA_PATH
    test_path = TEST_SQL_DATA_PATH
    
    if not os.path.exists(train_path):
        print("Preprocessed data not found. Running preprocessing...")
        from preprocess_sql import preprocess_sql_qa_pipeline
        preprocess_sql_qa_pipeline()
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_questions = train_df['cleaned_question'].tolist()
    train_answers = train_df['cleaned_answer'].tolist()
    test_questions = test_df['cleaned_question'].tolist()
    test_answers = test_df['cleaned_answer'].tolist()
    
    print(f"\nLoaded data:")
    print(f"  Train: {len(train_questions)} Q&A pairs")
    print(f"  Test:  {len(test_questions)} Q&A pairs")
    
    # Create vocabularies from ALL data
    all_questions = train_questions + test_questions
    all_answers = train_answers + test_answers
    
    (question_word2int, question_int2word,
     answer_word2int, answer_int2word) = create_qa_vocabularies(
        all_questions, all_answers
    )
    
    # Create dataloader (ONLY from training data)
    print("\nCreating training dataloader...")
    train_dataloader = create_qa_dataloader(
        train_questions, train_answers,
        question_word2int, answer_word2int,
        BATCH_SIZE, shuffle=True
    )
    
    # Create model
    print("\nCreating model...")
    encoder, decoder = create_qa_model(
        len(question_word2int), len(answer_word2int),
        EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DEVICE, DROPOUT
    )
    
    # Train model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    encoder, decoder = train_qa_model(
        encoder, decoder, train_dataloader,
        question_word2int, answer_word2int,
        num_epochs=NUM_EPOCHS
    )
    
    print("\n✅ Training completed!")
