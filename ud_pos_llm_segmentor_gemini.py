import os
import json
import random
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

# Import utility functions to read the UD dataset
from utils import read_conllu, untag

# Load environment variables and configure API
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found in environment variables.")

genai.configure(api_key=api_key)

# Model to use
gemini_model = 'gemini-2.0-flash-lite'

def create_segmentation_prompt() -> str:
    """
    Create a system prompt for the segmentation task.
    """
    return """# Universal Dependencies Tokenization

Your task is to segment sentences according to Universal Dependencies (UD) tokenization guidelines.

Key tokenization rules:
1. Separate all punctuation marks as individual tokens
2. Split contractions:
   - "don't" → ["Do", "n't"]
   - "I'm" → ["I", "'m"]
   - "can't" → ["ca", "n't"]
3. Split possessives:
   - "John's" → ["John", "'s"]
4. Keep hyphenated compounds as single tokens when they function as a unit
5. Numbers with internal punctuation are typically a single token (e.g., "3.14", "$50.00")
6. Handle special cases like URLs and email addresses as single tokens

Output a JSON array containing the segmented tokens. Follow the examples exactly.
"""

def create_examples() -> str:
    """
    Create few-shot examples for the segmentation task.
    """
    examples = [
        {
            "original": "Don't worry, I'll meet you at John's house at 3:30pm.",
            "segmented": ["Do", "n't", "worry", ",", "I", "'ll", "meet", "you", "at", "John", "'s", "house", "at", "3:30pm", "."]
        },
        {
            "original": "She said: \"I can't believe it!\" and walked away.",
            "segmented": ["She", "said", ":", "\"", "I", "ca", "n't", "believe", "it", "!", "\"", "and", "walked", "away", "."]
        },
        {
            "original": "The well-known CEO bought 2,500 shares for $47.50 each.",
            "segmented": ["The", "well-known", "CEO", "bought", "2,500", "shares", "for", "$", "47.50", "each", "."]
        }
    ]
    
    prompt_examples = ""
    for example in examples:
        prompt_examples += f"Original: {example['original']}\n"
        prompt_examples += f"Segmented: {json.dumps(example['segmented'])}\n\n"
    
    return prompt_examples

@sleep_and_retry
@limits(calls=15, period=60)
def segment_sentence(text: str) -> List[str]:
    """
    Segment a single sentence according to UD guidelines.
    
    Args:
        text: The original sentence to segment
        
    Returns:
        A list of tokens from the segmented sentence
    """
    # Create system prompt and examples
    system_prompt = create_segmentation_prompt()
    examples = create_examples()
    
    # Build the user prompt with the sentence to segment
    user_prompt = "Please segment this sentence according to Universal Dependencies guidelines:\n\n"
    user_prompt += examples
    user_prompt += f"Original: {text}\n"
    user_prompt += "Segmented: "
    
    # Send prompt to Gemini API
    client = genai.Client()
    response = client.models.generate_content(
        model=gemini_model,
        contents=[
            {"role": "system", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": user_prompt}]}
        ],
        config={
            'temperature': 0.0,  # For deterministic output
        },
    )
    
    if not response.text:
        raise ValueError("No response received from Gemini API")
    
    # Try to extract JSON array from the response
    response_text = response.text.strip()
    
    try:
        # Try direct JSON parsing
        tokens = json.loads(response_text)
        if isinstance(tokens, list):
            return tokens
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract array portion
        import re
        array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if array_match:
            try:
                tokens = json.loads(array_match.group(0))
                if isinstance(tokens, list):
                    return tokens
            except json.JSONDecodeError:
                pass
    
    # Fallback - return simple space-split tokens
    print(f"Warning: Could not parse JSON from response for sentence: {text[:30]}...")
    return text.split()

def load_ud_english_data(conllu_file_path: str, sample_size: int = None, random_seed: int = 42) -> Tuple[List[List[Tuple[str, str]]], List[str]]:
    """
    Load sentences from the UD English dataset.
    
    Args:
        conllu_file_path: Path to the CoNLL-U format file
        sample_size: Optional number of sentences to sample
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple containing tagged sentences and original sentences
    """
    print(f"Loading UD English data from {conllu_file_path}...")
    
    # Read the CoNLL-U file
    tagged_sentences, original_sentences = read_conllu(conllu_file_path)
    
    if sample_size and sample_size < len(tagged_sentences):
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Select a random sample of indices
        indices = random.sample(range(len(tagged_sentences)), sample_size)
        
        # Sample the sentences
        tagged_sentences = [tagged_sentences[i] for i in indices]
        original_sentences = [original_sentences[i] for i in indices]
        
        print(f"Sampled {sample_size} sentences from the dataset")
    else:
        print(f"Using all {len(tagged_sentences)} sentences from the dataset")
    
    return tagged_sentences, original_sentences

def create_sentence_token_map_from_ud(conllu_file_path: str, output_file: str, sample_size: int = None):
    """
    Create a JSON file mapping UD English sentences to their tokenized versions.
    
    Args:
        conllu_file_path: Path to the CoNLL-U format file
        output_file: Path to save the mapping
        sample_size: Optional number of sentences to sample
    """
    # Load sentences from the UD English dataset
    tagged_sentences, original_sentences = load_ud_english_data(conllu_file_path, sample_size)
    
    # Ensure we only process the requested number of sentences
    if sample_size and sample_size < len(tagged_sentences):
        tagged_sentences = tagged_sentences[:sample_size]
        original_sentences = original_sentences[:sample_size]
    
    sentence_map = []
    
    print(f"Processing {len(original_sentences)} sentences...")
    for i, original_sentence in enumerate(tqdm(original_sentences)):
        # Extract gold tokens from the tagged sentences
        gold_tokens = untag(tagged_sentences[i])
        
        try:
            # Get segmented tokens from the LLM
            predicted_tokens = segment_sentence(original_sentence)
            
            # Add to the mapping
            sentence_map.append({
                "original": original_sentence,
                "gold_tokens": gold_tokens,
                "predicted_tokens": predicted_tokens
            })
            
        except Exception as e:
            print(f"Error processing sentence: {original_sentence[:30]}... - {str(e)}")
            # Add the error case to the map with gold tokens
            sentence_map.append({
                "original": original_sentence,
                "gold_tokens": gold_tokens,
                "error": str(e)
            })
    
    # Save the mapping to a JSON file
    print(f"Saving mapping to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(sentence_map, f, indent=2)
    
    print(f"Successfully created sentence-to-token mapping for {len(sentence_map)} sentences.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    # File paths
    UD_ENGLISH_TEST = '../UD_English-EWT/en_ewt-ud-test.conllu'
    OUTPUT_MAP_FILE = "ud_sentence_tokens_map.json"
    
    # Optional: Sample size (set to None to process all)
    SAMPLE_SIZE = 200  # Process 200 sentences, adjust as needed
    
    # Create the sentence-to-token mapping using the UD English dataset
    create_sentence_token_map_from_ud(
        conllu_file_path=UD_ENGLISH_TEST,
        output_file=OUTPUT_MAP_FILE,
        sample_size=SAMPLE_SIZE
    )