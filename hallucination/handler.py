import os
from datetime import datetime

base_question_dir = "./reasoning_questions_hallucination"
log_dir = "./hallucination_logs"
output_dir = "./hallucination_results"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

question_subdirs = sorted(
    os.path.join(base_question_dir, d)
    for d in os.listdir(base_question_dir)
    if os.path.isdir(os.path.join(base_question_dir, d))
)

for q_subdir in question_subdirs:
    q_id = os.path.basename(q_subdir)
    question_path = os.path.join(q_subdir, "question.txt")
    result_dir = os.path.join(output_dir, q_id)
    log_path = os.path.join(log_dir, f"{q_id}.log")
    os.makedirs(result_dir, exist_ok=True)

    print(f"▶️ Running: {q_id}")
    os.system(
        f"python3 cot.py "
        f"--questions_dir {q_subdir} "
        f"--output_dir {result_dir} "
        f"--model llama3.2 "
        f"> {log_path} 2>&1"
    )
    print(f"✅ Done: {q_id} (log: {log_path})")
