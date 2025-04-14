import os
import argparse
from datasets import load_dataset

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract questions from HuggingFace dataset')
    parser.add_argument('--output_dir', type=str, default='./reasoning_questions', help='Directory to save question text files')
    parser.add_argument('--dataset', type=str, default='simplescaling/s1K', help='HuggingFace dataset to download')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--max_questions', type=int, default=None, help='Maximum number of questions to extract (None for all)')
    return parser.parse_args()

def extract_questions(dataset_name, split, output_dir, max_questions=None):
    """Extract questions from dataset and save them as text files."""
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)
    
    # Get dataset structure
    print(f"Dataset structure: {dataset.features}")
    
    # Determine the question field name (assuming it's likely to be called 'question', 'prompt', 'input', etc.)
    possible_question_fields = ['question', 'prompt', 'input', 'text', 'instruction']
    question_field = None
    
    for field in possible_question_fields:
        if field in dataset.features:
            question_field = field
            print(f"Using field '{field}' as the question source")
            break
    
    if not question_field:
        # Show available fields to choose from
        print(f"Could not determine question field. Available fields: {list(dataset.features.keys())}")
        question_field = input("Please enter the field name containing questions: ").strip()
    
    # Limit the number of questions if specified
    if max_questions is not None:
        dataset = dataset.select(range(min(max_questions, len(dataset))))
    
    # Process each example
    for i, example in enumerate(dataset):
        # Get the question text
        question_text = example[question_field]
        
        # Create a filename
        filename = f"question_{i+1:04d}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(question_text)
        
        if (i+1) % 10 == 0:
            print(f"Saved {i+1}/{len(dataset)} questions")
    
    print(f"Extracted {len(dataset)} questions to {output_dir}")

if __name__ == "__main__":
    args = setup_args()
    extract_questions(args.dataset, args.split, args.output_dir, args.max_questions)
    print("Question extraction complete!")
