# LLaMA 3.2 — Simulating GPT-4 Benchmarks, Reasoning, and Hallucination

This project evaluates the behavior of LLaMA 3.2 as a proxy for GPT-4, simulating how GPT-4 performs across reasoning-heavy, mathematical, and hallucination-prone scenarios. It uses [Ollama](https://ollama.com/) for running LLaMA 3.2 locally, and [Hugging Face Datasets](https://huggingface.co/datasets) for sourcing questions and prompts.

The project simulates chain-of-thought reasoning, runs multiple iterations per question, and observes hallucinations under adversarial pressure.

## Getting Started

### Requirements

- Python 3.8+
- [Ollama](https://ollama.com) installed and running
- Ollama model pulled:
  ```bash
  ollama pull llama3.2
  ```

### Installation

Install Python dependencies:
```bash
pip install datasets requests
```

### Directory Structure

```
.
├── reasoning/  
   |-- reasoning_ques              # Chain-of-thought reasoning questions
├── math/     
    ├── math_results/                    # Mathematics-focused questions
├── hallucination/   
     ├── hallucination_results/
     ├── hallucination_logs/       # JSON hallucination logs
               # Adversarial prompts to trigger hallucinations
        # JSON responses from LLaMA for reasoning tasks
           # JSON responses for math evaluation

├── reasoning_logs/            # Terminal output logs for reasoning
├── math_logs/                 # Logs for math runs
      # Logs capturing adversarial output
└── README.md
```

## Running Experiments

Make sure `ollama serve` is running in the background:
```bash
ollama serve
```

Then run the evaluation on any folder:
```bash
python cot.py \
  --model llama3.2 \
  --questions_dir ./hallucination/q01 \
  --output_dir ./hallucination_results/q01
```

## Methodology

- Each question is passed to LLaMA 3.2
- The output is recursively fed back into the model for 5 iterations
- Responses are stored in structured JSON files for analysis
- Terminal output is logged for reproducibility

## Datasets Used

Prompts are generated from:
- simplescaling/s1K
- GAIR/o1-journey
- Custom-authored adversarial prompts to trigger hallucinations and logical breakdowns

## Objectives

- **Benchmark**: Simulate GPT-4 behavior using LLaMA 3.2
- **Stress-test**: Push the model with logic traps, math puzzles, and paradoxes
- **Evaluate**: Observe hallucinations, inconsistencies, and overconfidence
- **Compare**: Align model behavior with known GPT-4 limitations from OpenAI's technical report.
