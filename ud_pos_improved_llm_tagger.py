import os
import sys
import json
import time
from typing import List, Dict, Union, Optional, Tuple
from difflib import SequenceMatcher

import nltk
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

from utils import read_conllu, untag
from schema import TaggedSentences, TokenPOS
from prompts import tagger_prompt, create_segmentation_prompt, create_examples

load_dotenv()
nltk.download('punkt_tab')

# Gemini model ID to use
gemini_model = 'gemini-2.0-flash-lite'

# --- Configure the Gemini API ---
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Warning: API key not found in environment variables.")
        print("   Please set the GOOGLE_API_KEY environment variable.")
        sys.exit(1)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    sys.exit(1)

# --- Segmentation Function ---
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
    user_prompt = system_prompt + "\n\n" 
    user_prompt += "Please segment this sentence according to Universal Dependencies guidelines:\n\n"
    user_prompt += examples
    user_prompt += f"Original: {text}\n"
    user_prompt += "Segmented: "
    
    # Send prompt to Gemini API
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=user_prompt, 
        config={
            'temperature': 0.0,
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


# --- POS Tagging Function ---
@sleep_and_retry
@limits(calls=15, period=60)
def tag_sentences_ud(
        texts_to_tag: Optional[Union[str, List[str]]] = None,
        tokens_to_tag: Optional[Union[List[str], List[List[str]]]] = None
    ) -> TaggedSentences:
    """
    Tags sentences with Universal Dependencies (UD) part-of-speech tags using the Gemini API.
    
    Args:
        texts_to_tag: A single string or a list of strings representing the text(s) to be tagged.
        tokens_to_tag: A list of tokens (or a list of lists of tokens) to be tagged.
        
    Returns:
        TaggedSentences: A data structure containing the tagged sentences with UD part-of-speech tags.
    """
    if texts_to_tag is None and tokens_to_tag is None:
        raise ValueError("Either texts_to_tag or tokens_to_tag must be provided.")
    
    if texts_to_tag is not None:
        if isinstance(texts_to_tag, str) and isinstance(tokens_to_tag, str):
            texts_to_tag = [texts_to_tag]
            if tokens_to_tag:
                tokens_to_tag = [tokens_to_tag]
        elif isinstance(texts_to_tag, list) and (not isinstance(tokens_to_tag, list) or not isinstance(tokens_to_tag[0], list)):
            raise TypeError("If texts_to_tag is a list, tokens_to_tag must also be a list of lists.")
        
        if not tokens_to_tag:
            tokens_to_tag = [nltk.word_tokenize(text) for text in texts_to_tag]

    # Construct the prompt
    prompt = tagger_prompt(tokens_to_tag)

    # Send prompt to the Gemini API
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'temperature': 0.0,
        },
    )

    if response.text is None:
        raise ValueError("No response from Gemini API. Please check your API key and network connection.")
    
    resp_dict = json.loads(response.text)
    res = TaggedSentences(**resp_dict)
    return res


# --- Integrated Pipeline Function ---
def segment_and_tag_sentences(sentences: List[str], batch_size: int = 1) -> List[Dict]:
    """
    Process sentences through both segmentation and tagging in a pipeline.
    
    Args:
        sentences: List of sentences to process
        batch_size: Number of sentences to process in each tagging batch
        
    Returns:
        List of dictionaries containing original sentences, tokens, and POS tags
    """
    results = []
    
    # Process each sentence
    print(f"Processing {len(sentences)} sentences...")
    for i, original_sentence in enumerate(tqdm(sentences, desc="Segmenting and tagging")):
        try:
            # Step 1: Segment the sentence
            segmented_tokens = segment_sentence(original_sentence)
            
            # Step 2: Create the tokenized version for tagging
            tokenized_text = " ".join(segmented_tokens)
            
            # Step 3: Tag the tokenized sentence
            tagged_result = tag_sentences_ud([tokenized_text], [segmented_tokens])
            
            # Check if we have valid tags
            if tagged_result.sentences and tagged_result.sentences[0].sentence_tags:
                pos_tags = [
                    {
                        "token": tag.token,
                        "pos_tag": tag.pos_tag
                    } for tag in tagged_result.sentences[0].sentence_tags
                ]
                
                # Create the result entry
                result_entry = {
                    "original": original_sentence,
                    "tokens": segmented_tokens,
                    "tags": pos_tags
                }
                results.append(result_entry)
            else:
                print(f"⚠️ Warning: No tags returned for sentence {i+1}: {original_sentence[:30]}...")
                results.append({
                    "original": original_sentence,
                    "tokens": segmented_tokens,
                    "error": "No tags returned by the LLM"
                })
                
        except Exception as e:
            print(f"Error processing sentence {i+1}: {str(e)}")
            results.append({
                "original": original_sentence,
                "error": str(e)
            })
            time.sleep(1)  # Give API a short break on errors
    
    return results


def evaluate_tags(results: List[Dict], reference_data: Optional[List[Dict]] = None) -> Dict:
    """
    Evaluate the accuracy of the POS tagging results.
    
    Args:
        results: List of tagged sentence results
        reference_data: Optional reference data with correct tags
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "total_sentences": len(results),
        "successful_sentences": sum(1 for r in results if "error" not in r),
        "total_tokens": sum(len(r.get("tokens", [])) for r in results if "tokens" in r),
    }
    
    # If we have reference data, compute accuracy
    if reference_data:
        correct_tags = 0
        total_comparable_tags = 0
        
        for res, ref in zip(results, reference_data):
            if "error" in res or "tokens" not in res:
                continue
                
            # Extract reference tokens and tags
            ref_tokens = ref.get("gold_tokens", [])
            ref_tags = ref.get("gold_tags", [])
            
            # Extract predicted tokens and tags
            pred_tokens = res.get("tokens", [])
            pred_tags = [tag.get("pos_tag") for tag in res.get("tags", [])]
            
            # Compare only if we have matching token counts
            if len(ref_tokens) == len(pred_tokens):
                total_comparable_tags += len(ref_tokens)
                for i, (ref_token, pred_token) in enumerate(zip(ref_tokens, pred_tokens)):
                    if ref_token == pred_token and i < len(ref_tags) and i < len(pred_tags):
                        if ref_tags[i] == pred_tags[i]:
                            correct_tags += 1
        
        metrics["tag_accuracy"] = correct_tags / total_comparable_tags if total_comparable_tags > 0 else 0
    
    return metrics



def process_conllu(conllu_file_path: str, output_file: str, sample_size: Optional[int] = None):
    """
    Process sentences from a CoNLL-U file through the integrated pipeline.
    
    Args:
        conllu_file_path: Path to the CoNLL-U format file
        output_file: Path to save the tagged results
        sample_size: Optional number of sentences to sample
    """
    # Load sentences from the UD dataset
    print(f"Loading data from {conllu_file_path}...")
    tagged_sentences, original_sentences = read_conllu(conllu_file_path)
    
    # Apply sample size limit if specified
    if sample_size and sample_size < len(original_sentences):
        tagged_sentences = tagged_sentences[:sample_size]
        original_sentences = original_sentences[:sample_size]
        print(f"Using first {sample_size} sentences")
    
    # Extract gold standard tokens and tags
    gold_data = []
    for i, sentence in enumerate(tagged_sentences):
        gold_tokens = untag(sentence)
        gold_tags = [tag for _, tag in sentence]
        
        gold_data.append({
            "original": original_sentences[i],
            "gold_tokens": gold_tokens,
            "gold_tags": gold_tags
        })
    
    # Process sentences through the pipeline
    results = segment_and_tag_sentences(original_sentences)
    
    # Evaluate against gold standard
    metrics = evaluate_tags(results, gold_data)
    
    # Add gold standard data to results for comparison
    for i, result in enumerate(results):
        if i < len(gold_data):
            result["gold_tokens"] = gold_data[i]["gold_tokens"]
            result["gold_tags"] = gold_data[i]["gold_tags"]
    
    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": metrics,
            "results": results
        }, f, indent=2)
    
    print(f"Successfully processed {len(results)} sentences.")
    print(f"Results saved to {output_file}")
    
    # Print evaluation metrics
    print("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    process_conllu(
        conllu_file_path="../UD_English-EWT/en_ewt-ud-test.conllu",
        output_file="improved_ud_sentences_tagged.json",
        sample_size=200
    )