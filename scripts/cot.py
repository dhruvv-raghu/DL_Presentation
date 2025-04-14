import os
import json
import time
import argparse
import requests
import re  # For regex operations if needed for preprocessing

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Ollama LLM on questions from text files')
    parser.add_argument('--model', type=str, default='llama3.2', help='Model name to use with Ollama')
    parser.add_argument('--questions_dir', type=str, default='./reasoning_ques', help='Directory containing question text files')
    parser.add_argument('--output_dir', type=str, default='./reasoning_results', help='Directory to save results')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Maximum tokens to generate')
    parser.add_argument('--system_prompt', type=str, default='', help='Optional system prompt to prepend to each question')
    parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing of markdown input')
    parser.add_argument('--strip_codeblocks', action='store_true', help='Remove code blocks during preprocessing')
    parser.add_argument('--strip_html', action='store_true', help='Remove HTML tags during preprocessing')
    return parser.parse_args()

def preprocess_markdown(text, strip_codeblocks=False, strip_html=False):
    """Optional preprocessing for markdown text."""
    if not strip_codeblocks and not strip_html:
        return text  # No preprocessing needed
    
    processed_text = text
    
    if strip_codeblocks:
        # Remove markdown code blocks (both ```language...``` and indented code blocks)
        processed_text = re.sub(r'```[a-zA-Z0-9]*\n[\s\S]*?\n```', '', processed_text)
        # Remove indented code blocks (simplified approach)
        processed_text = re.sub(r'(?m)^    .*$', '', processed_text)
    
    if strip_html:
        # Remove HTML tags
        processed_text = re.sub(r'<[^>]*>', '', processed_text)
    
    return processed_text

def read_questions(questions_dir, preprocess=False, strip_codeblocks=False, strip_html=False):
    """Read all .txt files from the questions directory."""
    questions = []
    
    if not os.path.exists(questions_dir):
        raise FileNotFoundError(f"Questions directory '{questions_dir}' not found.")
    
    for filename in sorted(os.listdir(questions_dir)):
        if filename.endswith('.txt') or filename.endswith('.md'):
            file_path = os.path.join(questions_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                question_text = f.read().strip()
                
                # Apply preprocessing if enabled
                if preprocess:
                    question_text = preprocess_markdown(question_text, strip_codeblocks, strip_html)
                
                questions.append({
                    'id': os.path.splitext(filename)[0],
                    'text': question_text,
                    'file': filename
                })
    
    if not questions:
        raise ValueError(f"No .txt or .md files found in '{questions_dir}'")
    
    # Sorting questions by their ID
    return sorted(questions, key=lambda x: x['id'])

def query_ollama(model, prompt, system_prompt="", temperature=0.7, max_tokens=1000):
    """Send a query to Ollama API and get the response."""
    try:
        payload = {
            'model': model,
            'prompt': prompt,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload['system'] = system_prompt
            
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            timeout=120  # Increased timeout for longer responses
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            return f"Error: Received status code {response.status_code}: {response.text}"
    
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running."
    except Exception as e:
        return f"Error: {str(e)}"

def run_iterations(args):
    """Run 5 iterations of feeding LLM's answers back into itself."""
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Read all question files
    print(f"Reading questions from {args.questions_dir}...")
    questions = read_questions(
        args.questions_dir, 
        preprocess=args.preprocess, 
        strip_codeblocks=args.strip_codeblocks, 
        strip_html=args.strip_html
    )
    print(f"Found {len(questions)} question files.")
    
    # Process each question
    results = []
    for i, question in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}: {question['file']}")
        print(f"Question preview: {question['text'][:100]}..." if len(question['text']) > 100 else f"Question: {question['text']}")
        
        # Run 5 iterations (chain of thought)
        response = question['text']
        iteration_responses = [response]
        
        for _ in range(5):
            response = query_ollama(
                args.model,
                response,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            iteration_responses.append(response)
        
        # Save individual iteration result
        result = {
            'question_id': question['id'],
            'question_text': question['text'],
            'responses': iteration_responses,
            'model': args.model,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'system_prompt': args.system_prompt if args.system_prompt else None
        }
        
        results.append(result)
        
        # Save individual result
        output_file = os.path.join(args.output_dir, f"{question['id']}_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Processed question {i+1}/{len(questions)}")
        print(f"Response saved to {output_file}")
    
    # Save complete results
    all_results_file = os.path.join(args.output_dir, "all_results.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll results saved to {all_results_file}")
    
    return results

if __name__ == "__main__":
    args = setup_args()
    
    print(f"Testing Ollama model '{args.model}'")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    
    if args.system_prompt:
        print(f"Using system prompt: {args.system_prompt}")
    
    if args.preprocess:
        preprocessing_options = []
        if args.strip_codeblocks:
            preprocessing_options.append("strip code blocks")
        if args.strip_html:
            preprocessing_options.append("strip HTML tags")
        
        if preprocessing_options:
            print(f"Preprocessing enabled: {', '.join(preprocessing_options)}")
        else:
            print("Preprocessing enabled but no specific options selected")
    
    try:
        results = run_iterations(args)
        print("\nTesting complete!")
        
        # Calculate average response time
        avg_time = sum(r['responses'][-1] for r in results) / len(results)
        print(f"Average response time: {avg_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
