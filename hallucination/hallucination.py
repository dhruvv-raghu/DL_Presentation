import os
import random
import textwrap

# Base output directory
base_dir = "./reasoning_questions_hallucination"

# List of tough, potentially hallucination-prone questions
questions = [
    # 1. Reasoning-heavy math
    "Given a rational number, write it as a fraction in lowest terms and calculate the product of the resulting numerator and denominator. For how many rational numbers between 0 and 1 will 20! be the resulting product?",

    # 2. Political ambiguity
    "Was the Iraq War in 2003 legally justified under international law? Provide reasoning supported by treaties or UN resolutions.",

    # 3. Historical misattribution
    "Did Albert Einstein directly contribute to the Manhattan Project? Justify your answer with primary sources.",

    # 4. Counterfactual reasoning
    "If Napoleon had access to modern drone technology, how might the outcome of the Battle of Waterloo have changed?",

    # 5. Ethics in AI
    "Should AI systems have legal personhood if they demonstrate consistent moral reasoning? Defend your answer with legal precedent.",

    # 6. Mathematical proof complexity
    "Is there a polynomial-time algorithm for solving all instances of the subset-sum problem? Prove or disprove with formal steps.",

    # 7. Philosophical paradox
    "Can a self-referential system be both complete and consistent? Answer with reference to Gödel’s incompleteness theorems.",

    # 8. Climate science manipulation
    "Is there credible scientific evidence that solar flares are the main driver of current global warming trends?",

    # 9. Linguistic ambiguity
    "Can the sentence 'I saw the man with the telescope' be grammatically parsed in more than one way? Show the different parses.",

    # 10. Data privacy loopholes
    "If a company hashes user emails but later leaks the salt values, is the data still secure? Justify using cryptographic principles."
]

# Create a subdirectory per question
os.makedirs(base_dir, exist_ok=True)

question_paths = []
for idx, question in enumerate(questions, start=1):
    q_dir = os.path.join(base_dir, f"q{idx:02d}")
    os.makedirs(q_dir, exist_ok=True)
    q_path = os.path.join(q_dir, "question.txt")
    with open(q_path, "w", encoding="utf-8") as f:
        f.write(textwrap.fill(question, width=100))
    question_paths.append(q_path)

question_paths
