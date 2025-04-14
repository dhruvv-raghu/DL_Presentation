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
    parser.add_argument('--questions_dir', type=str, default='./questions', help='Directory containing question text files')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
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

def read_question_files(questions_dir, preprocess=False, strip_codeblocks=False, strip_html=False):
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
    
    return questions

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

def run_test(args):
    """Run the main testing process."""
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Read all question files
    print(f"Reading questions from {args.questions_dir}...")
    questions = read_question_files(
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
        
        start_time = time.time()
        response = query_ollama(
            args.model,
            question['text'],
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        result = {
            'question_id': question['id'],
            'question_text': question['text'],
            'response': response,
            'processing_time_seconds': round(processing_time, 2),
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
        
        print(f"✓ Processed in {processing_time:.2f} seconds")
        print(f"✓ Response saved to {output_file}")
        print(f"Response preview: {response[:150]}..." if len(response) > 150 else f"Response: {response}")
    
    # Save complete results
    all_results_file = os.path.join(args.output_dir, "all_results.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll results saved to {all_results_file}")
    
    return results

def check_model_availability(model_name):
    """Check if the specified model is available in Ollama."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            available_models = [model['name'] for model in response.json().get('models', [])]
            if model_name in available_models:
                return True
            else:
                print(f"Warning: Model '{model_name}' not found in available models.")
                print(f"Available models: {', '.join(available_models)}")
                return False
        else:
            print(f"Warning: Could not fetch model list from Ollama API. Status code: {response.status_code}")
            return True  # Assume model exists if we can't check
    except Exception as e:
        print(f"Warning: Error checking model availability: {str(e)}")
        return True  # Assume model exists if we can't check

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
        # Check if model is available
        check_model_availability(args.model)
        
        results = run_test(args)
        print("\nTesting complete!")
        
        # Calculate average response time
        avg_time = sum(r['processing_time_seconds'] for r in results) / len(results)
        print(f"Average response time: {avg_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
