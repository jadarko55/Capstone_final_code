"""
SQL Knowledge Tracing - Data Simulation
Based on: Deep Knowledge Tracing (Piech et al., 2015)

Simulates virtual students learning SQL concepts using Item Response Theory (IRT).
Generates interaction sequences suitable for training a DKT LSTM model.
"""

import numpy as np
import pandas as pd
import json
import os

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

# SQL concept taxonomy (8 concepts)
CONCEPTS = {
    0: "SELECT/FROM Basics",
    1: "WHERE and Filtering",
    2: "JOINs",
    3: "GROUP BY and Aggregation",
    4: "Subqueries",
    5: "NULL Handling",
    6: "ORDER BY and LIMIT",
    7: "String and Date Functions",
}

NUM_CONCEPTS      = len(CONCEPTS)       # 8
EXERCISES_PER_CONCEPT = 10              # 10 exercises per concept = 80 total
NUM_EXERCISES     = NUM_CONCEPTS * EXERCISES_PER_CONCEPT  # 80

NUM_STUDENTS      = 2000                # total virtual students
SEQ_LENGTH        = 40                  # interactions per student

GUESS_PROB        = 0.25                # c in IRT (random guess probability)
LEARN_INCREMENT   = 0.1                 # skill gain after correct answer
FORGET_DECREMENT  = 0.05               # small skill penalty after wrong answer

# Initial skill drawn from N(mean, std) — negative mean = beginners
SKILL_MEAN        = -0.5
SKILL_STD         = 1.0

# Exercise difficulties spread uniformly across [-2, 2] within each concept
DIFFICULTY_LOW    = -2.0
DIFFICULTY_HIGH   =  2.0


# ─────────────────────────────────────────────
# 2. GENERATE EXERCISES
# ─────────────────────────────────────────────

def generate_exercises():
    """
    Create exercise definitions.
    Each exercise has:
      - exercise_id  : unique int (0 to NUM_EXERCISES-1)
      - concept_id   : which SQL concept it tests
      - difficulty   : float in [DIFFICULTY_LOW, DIFFICULTY_HIGH]
      - label        : human-readable string
    """
    exercises = []
    for concept_id in range(NUM_CONCEPTS):
        difficulties = np.linspace(DIFFICULTY_LOW, DIFFICULTY_HIGH, EXERCISES_PER_CONCEPT)
        for i, diff in enumerate(difficulties):
            ex_id = concept_id * EXERCISES_PER_CONCEPT + i
            exercises.append({
                "exercise_id":  ex_id,
                "concept_id":   concept_id,
                "concept_name": CONCEPTS[concept_id],
                "difficulty":   round(float(diff), 3),
                "label":        f"{CONCEPTS[concept_id]} - Q{i+1} (diff={diff:.1f})",
            })
    return exercises


# ─────────────────────────────────────────────
# 3. IRT PROBABILITY
# ─────────────────────────────────────────────

def irt_probability(skill: float, difficulty: float, c: float = GUESS_PROB) -> float:
    """
    3-parameter IRT model:
        P(correct | skill α, difficulty β) = c + (1-c) / (1 + exp(β - α))

    - Higher skill  → higher probability
    - Higher difficulty → lower probability
    - c is the floor (guessing)
    """
    return c + (1 - c) / (1 + np.exp(difficulty - skill))


# ─────────────────────────────────────────────
# 4. SIMULATE STUDENTS
# ─────────────────────────────────────────────

def simulate_students(exercises: list) -> list:
    """
    Simulate NUM_STUDENTS students each answering SEQ_LENGTH exercises.

    Returns a list of student interaction sequences.
    Each sequence is a list of dicts:
        { student_id, step, exercise_id, concept_id, difficulty,
          skill_before, p_correct, correct }
    """
    exercise_array = np.array([(e["concept_id"], e["difficulty"]) for e in exercises])
    all_interactions = []

    for student_id in range(NUM_STUDENTS):

        # Each student starts with a random skill per concept
        skills = np.random.normal(SKILL_MEAN, SKILL_STD, NUM_CONCEPTS)

        student_log = []

        for step in range(SEQ_LENGTH):

            # Randomly sample an exercise (uniform — no fixed sequence)
            ex_id = np.random.randint(0, NUM_EXERCISES)
            concept_id = int(exercise_array[ex_id, 0])
            difficulty  = float(exercise_array[ex_id, 1])

            skill_before = float(skills[concept_id])
            p_correct    = irt_probability(skill_before, difficulty)
            correct      = int(np.random.binomial(1, p_correct))

            student_log.append({
                "student_id":   student_id,
                "step":         step,
                "exercise_id":  ex_id,
                "concept_id":   concept_id,
                "concept_name": CONCEPTS[concept_id],
                "difficulty":   round(difficulty, 3),
                "skill_before": round(skill_before, 4),
                "p_correct":    round(p_correct, 4),
                "correct":      correct,
            })

            # Learning update
            if correct:
                skills[concept_id] += LEARN_INCREMENT
            else:
                skills[concept_id] -= FORGET_DECREMENT

        all_interactions.append(student_log)

        if (student_id + 1) % 500 == 0:
            print(f"  Simulated {student_id + 1}/{NUM_STUDENTS} students...")

    return all_interactions


# ─────────────────────────────────────────────
# 5. FORMAT FOR DKT INPUT
# ─────────────────────────────────────────────

def format_for_dkt(all_interactions: list, num_exercises: int) -> list:
    """
    Convert raw interaction logs into DKT-ready sequences.

    Each timestep input is a one-hot vector of length 2 * NUM_EXERCISES:
        - Indices [0 .. NUM_EXERCISES-1]          : exercise answered CORRECTLY
        - Indices [NUM_EXERCISES .. 2*NUM_EXERCISES-1] : exercise answered INCORRECTLY

    Target output at each timestep is the exercise_id answered at t+1.
    This matches the paper's encoding exactly.

    Returns a list of dicts, one per student:
        {
          "student_id": int,
          "inputs":     list of exercise_interaction_id   (int, range 0..2M-1)
          "targets":    list of exercise_id               (int, range 0..M-1)
          "corrects":   list of 0/1
        }
    """
    dkt_sequences = []

    for student_log in all_interactions:
        inputs  = []
        targets = []
        corrects = []

        for interaction in student_log:
            ex_id   = interaction["exercise_id"]
            correct = interaction["correct"]

            # One-hot index: correct → ex_id, incorrect → ex_id + NUM_EXERCISES
            input_idx = ex_id if correct else ex_id + num_exercises
            inputs.append(input_idx)
            targets.append(ex_id)
            corrects.append(correct)

        dkt_sequences.append({
            "student_id": student_log[0]["student_id"],
            "inputs":     inputs,    # x_t  (what they answered and how)
            "targets":    targets,   # q_t+1 (which exercise comes next)
            "corrects":   corrects,  # a_t+1 (ground truth for loss)
        })

    return dkt_sequences


# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(dkt_sequences: list, train_ratio: float = 0.8):
    """80/20 train-test split (student-level, not interaction-level)."""
    np.random.shuffle(dkt_sequences)
    split = int(len(dkt_sequences) * train_ratio)
    return dkt_sequences[:split], dkt_sequences[split:]


# ─────────────────────────────────────────────
# 7. SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_outputs(exercises, all_interactions, train_seqs, test_seqs, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Exercise definitions
    ex_df = pd.DataFrame(exercises)
    ex_df.to_csv(os.path.join(out_dir, "exercises.csv"), index=False)

    # Full flat interaction log (useful for EDA)
    flat = [item for student in all_interactions for item in student]
    flat_df = pd.DataFrame(flat)
    flat_df.to_csv(os.path.join(out_dir, "interactions_full.csv"), index=False)

    # DKT-ready sequences
    with open(os.path.join(out_dir, "train_sequences.json"), "w") as f:
        json.dump(train_seqs, f)
    with open(os.path.join(out_dir, "test_sequences.json"), "w") as f:
        json.dump(test_seqs, f)

    # Summary stats
    flat_df_summary = flat_df.groupby("concept_name")["correct"].agg(
        total="count", correct_count="sum"
    )
    flat_df_summary["accuracy"] = (
        flat_df_summary["correct_count"] / flat_df_summary["total"]
    ).round(3)
    flat_df_summary.to_csv(os.path.join(out_dir, "concept_accuracy_summary.csv"))

    print(f"\n✓ Saved all outputs to: {out_dir}")
    print(f"  exercises.csv              — {len(exercises)} exercises")
    print(f"  interactions_full.csv      — {len(flat):,} total interactions")
    print(f"  train_sequences.json       — {len(train_seqs)} students")
    print(f"  test_sequences.json        — {len(test_seqs)} students")
    print(f"  concept_accuracy_summary.csv\n")
    print(flat_df_summary.to_string())


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SQL DKT Data Simulation")
    print("=" * 55)
    print(f"  Students:           {NUM_STUDENTS}")
    print(f"  Concepts:           {NUM_CONCEPTS}")
    print(f"  Exercises total:    {NUM_EXERCISES}")
    print(f"  Sequence length:    {SEQ_LENGTH}")
    print(f"  Total interactions: {NUM_STUDENTS * SEQ_LENGTH:,}")
    print("=" * 55 + "\n")

    print("Step 1/4 — Generating exercises...")
    exercises = generate_exercises()

    print("Step 2/4 — Simulating students...")
    all_interactions = simulate_students(exercises)

    print("\nStep 3/4 — Formatting for DKT input...")
    dkt_sequences = format_for_dkt(all_interactions, NUM_EXERCISES)
    train_seqs, test_seqs = split_data(dkt_sequences)
    print(f"  Train: {len(train_seqs)} students | Test: {len(test_seqs)} students")

    print("\nStep 4/4 — Saving outputs...")
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    save_outputs(exercises, all_interactions, train_seqs, test_seqs,
                 out_dir=str(project_root / "data" / "sql_dkt_data"))

    print("\n✓ Simulation complete.")
    print("\nNext step: load train_sequences.json into your DKT LSTM model.")
    print("  Each sequence has keys: student_id, inputs, targets, corrects")
    print(f"  Input vector size (2M): {2 * NUM_EXERCISES}")
    print(f"  Output vector size (M): {NUM_EXERCISES}")


if __name__ == "__main__":
    main()
