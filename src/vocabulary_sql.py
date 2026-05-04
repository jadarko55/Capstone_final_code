"""Vocabulary building for SQL question-answer pairs."""
from collections import Counter
from config_sql import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MAX_VOCAB_SIZE

def tokenize_text(text):
    """Simple word tokenization."""
    if not isinstance(text, str):
        return []
    return text.lower().split()

def create_qa_vocabularies(questions, answers, max_vocab_size=MAX_VOCAB_SIZE):
    """Create separate vocabularies for questions and answers."""
    print("\n" + "="*80)
    print("BUILDING VOCABULARIES")
    print("="*80)
    
    # Count word frequencies
    question_word_counts = Counter()
    answer_word_counts = Counter()
    
    for question in questions:
        tokens = tokenize_text(question)
        question_word_counts.update(tokens)
    
    for answer in answers:
        tokens = tokenize_text(answer)
        answer_word_counts.update(tokens)
    
    print(f"\nUnique question words (before filtering): {len(question_word_counts)}")
    print(f"Unique answer words (before filtering): {len(answer_word_counts)}")
    
    # Create vocabularies with most common words
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    
    # Question vocabulary
    question_vocab = special_tokens.copy()
    most_common_q = [word for word, _ in question_word_counts.most_common(max_vocab_size - len(special_tokens))]
    question_vocab.extend(most_common_q)
    
    # Answer vocabulary
    answer_vocab = special_tokens.copy()
    most_common_a = [word for word, _ in answer_word_counts.most_common(max_vocab_size - len(special_tokens))]
    answer_vocab.extend(most_common_a)
    
    # Create mappings
    question_word2int = {word: idx for idx, word in enumerate(question_vocab)}
    question_int2word = {idx: word for word, idx in question_word2int.items()}
    
    answer_word2int = {word: idx for idx, word in enumerate(answer_vocab)}
    answer_int2word = {idx: word for word, idx in answer_word2int.items()}
    
    print(f"\nFinal question vocabulary size: {len(question_word2int)}")
    print(f"Final answer vocabulary size: {len(answer_word2int)}")
    print(f"\nSpecial tokens: {special_tokens}")
    
    # Calculate coverage
    q_total_words = sum(question_word_counts.values())
    q_covered_words = sum(count for word, count in question_word_counts.items() 
                         if word in question_word2int)
    q_coverage = (q_covered_words / q_total_words * 100) if q_total_words > 0 else 0
    
    a_total_words = sum(answer_word_counts.values())
    a_covered_words = sum(count for word, count in answer_word_counts.items() 
                         if word in answer_word2int)
    a_coverage = (a_covered_words / a_total_words * 100) if a_total_words > 0 else 0
    
    print(f"\nVocabulary Coverage:")
    print(f"  Questions: {q_coverage:.2f}% of all word occurrences")
    print(f"  Answers: {a_coverage:.2f}% of all word occurrences")
    
    return (question_word2int, question_int2word, 
            answer_word2int, answer_int2word)

if __name__ == "__main__":
    import pandas as pd
    from config_sql import TRAIN_SQL_DATA_PATH
    
    # Test vocabulary creation
    train_df = pd.read_csv(TRAIN_SQL_DATA_PATH)
    questions = train_df['cleaned_question'].tolist()
    answers = train_df['cleaned_answer'].tolist()
    
    (q_word2int, q_int2word, 
     a_word2int, a_int2word) = create_qa_vocabularies(questions, answers)
    
    print("\nSample question tokens:", list(q_word2int.keys())[:10])
    print("Sample answer tokens:", list(a_word2int.keys())[:10])
