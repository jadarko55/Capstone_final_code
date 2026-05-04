"""Inference utilities for question-answer generation."""
import torch
from vocabulary_sql import tokenize_text
from config_sql import DEVICE, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN

def generate_answer(encoder, decoder, question, question_word2int, 
                   answer_word2int, answer_int2word, max_length=200, temperature=0.8):
    """Generate answer from question with improved decoding."""
    encoder.eval()
    decoder.eval()
    
    with torch.inference_mode():
        # Tokenize and encode question
        tokens = tokenize_text(question)
        input_tensor = torch.tensor(
            [question_word2int.get(word, question_word2int[UNK_TOKEN]) for word in tokens]
            + [question_word2int.get(EOS_TOKEN, question_word2int[UNK_TOKEN])], 
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)
        
        # Encode question
        _, encoder_hidden, encoder_cell = encoder(input_tensor)
        
        # Initialize decoder
        decoder_input = torch.tensor(
            [answer_word2int[SOS_TOKEN]],
            dtype=torch.long
        ).view(1, 1).to(DEVICE)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        
        # Generate answer
        generated_tokens = []
        unk_count = 0
        max_unk = 3  # Stop if we get too many UNKs in a row
        
        for _ in range(max_length):
            logits, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell
            )
            
            # Apply temperature for more diverse outputs
            logits = logits.squeeze(1) / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Get next token
            next_token = torch.argmax(probs, dim=-1)
            token_id = next_token.item()
            
            # Stop conditions
            if token_id in [answer_word2int.get(EOS_TOKEN, -1), answer_word2int.get(PAD_TOKEN, -1)]:
                break
            
            # Track unknown tokens
            if token_id == answer_word2int.get(UNK_TOKEN, -1):
                unk_count += 1
                if unk_count >= max_unk:
                    break
            else:
                unk_count = 0  # Reset counter
                word = answer_int2word.get(token_id, UNK_TOKEN)
                if word != UNK_TOKEN:
                    generated_tokens.append(word)
            
            decoder_input = next_token.view(1,1)
        
        return ' '.join(generated_tokens) if generated_tokens else "I don't have enough information to answer that."

def batch_generate_answers(encoder, decoder, questions, question_word2int,
                          answer_word2int, answer_int2word, max_length=200):
    """Generate answers for multiple questions."""
    answers = []
    for question in questions:
        answer = generate_answer(
            encoder, decoder, question,
            question_word2int, answer_word2int, answer_int2word,
            max_length
        )
        answers.append(answer)
    return answers

if __name__ == "__main__":
    import pandas as pd
    from model_sql import create_qa_model
    from config_sql import EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, SQL_MODEL_PATH, DROPOUT, TEST_SQL_DATA_PATH
    
    print("="*80)
    print("QUESTION-ANSWER GENERATION TEST")
    print("="*80)
    
    # Load test data
    test_df = pd.read_csv(TEST_SQL_DATA_PATH)
    test_questions = test_df['cleaned_question'].tolist()
    test_answers = test_df['cleaned_answer'].tolist()
    
    # Load model and vocabularies
    print(f"\nLoading model from {SQL_MODEL_PATH}...")
    checkpoint = torch.load(SQL_MODEL_PATH, weights_only=False, map_location=DEVICE)
    
    question_word2int = checkpoint['question_word2int']
    question_int2word = checkpoint['question_int2word']
    answer_word2int = checkpoint['answer_word2int']
    answer_int2word = checkpoint['answer_int2word']
    
    # Create model
    encoder, decoder = create_qa_model(
        len(question_word2int), len(answer_word2int),
        EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DEVICE, DROPOUT
    )
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    print("✅ Model loaded successfully!\n")
    
    # Test predictions
    num_examples = min(5, len(test_questions))
    
    for i in range(num_examples):
        question = test_questions[i]
        true_answer = test_answers[i]
        
        print(f"\nExample {i+1}:")
        print(f"Question: {question[:150]}...")
        print(f"\nTrue Answer: {true_answer[:150]}...")
        
        generated_answer = generate_answer(
            encoder, decoder, question,
            question_word2int, answer_word2int, answer_int2word
        )
        
        print(f"\nGenerated Answer: {generated_answer[:150]}...")
        print("-" * 80)
    
    print("\n✅ Inference test completed!")
