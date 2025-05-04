# --- Imports ---
import os
import sys
import json
from typing import List, Union, Optional
from difflib import SequenceMatcher

import nltk
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
import time
from functools import wraps
from ratelimit import limits, sleep_and_retry

from utils import read_conllu
from schema import TaggedSentences, TokenPOS
from prompts import tagger_prompt


load_dotenv()
nltk.download('punkt_tab')

# gemini model id to use
gemini_model = 'gemini-2.0-flash-lite'


# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
        print("   Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")
        sys.exit(1)

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    sys.exit(1)


# --- Function to Perform POS Tagging ---
@sleep_and_retry
@limits(calls=15, period=60)
def tag_sentences_ud(
        texts_to_tag: Optional[Union[str, List[str]]] = None,
        tokens_to_tag: Optional[Union[List[str], List[List[str]]]] = None
    ) -> TaggedSentences:
    """
    Tags sentences with Universal Dependencies (UD) part-of-speech tags using the Gemini API.
    Args:
        texts_to_tag (Optional[Union[str, List[str]]]): 
            A single string or a list of strings representing the text(s) to be tagged. 
            If provided, the function will tokenize the text(s) if `tokens_to_tag` is not supplied.
        tokens_to_tag (Optional[Union[List[str], List[List[str]]]]): 
            A list of tokens (or a list of lists of tokens) to be tagged.
            If provided, the function will use these tokens directly without tokenizing `texts_to_tag`.
            If not provided, the function will tokenize `texts_to_tag` using NLTK's word tokenizer.
            If provided with `texts_to_tag`, they must match the structure.
    Returns:
        TaggedSentences: 
            A data structure containing the tagged sentences with Universal Dependencies (UD) part-of-speech tags.
    Raises:
        ValueError: 
            If neither `texts_to_tag` nor `tokens_to_tag` is provided, or if the Gemini API response is empty.
        TypeError: 
            If the types of `texts_to_tag` and `tokens_to_tag` are incompatible.
    """
    if texts_to_tag is None and tokens_to_tag is None:
        raise ValueError("Either texts_to_tag or tokens_to_tag must be provided.")
    
    if texts_to_tag is not None:
        if isinstance(texts_to_tag, str) and isinstance(tokens_to_tag, str):
            # If a single string is provided, convert it to a list
            texts_to_tag = [texts_to_tag]
            if tokens_to_tag:
                tokens_to_tag = [tokens_to_tag]
        elif isinstance(texts_to_tag, list) and (not isinstance(tokens_to_tag, list) or not isinstance(tokens_to_tag[0], list)):
            # If the types of texts_to_tag and tokens_to_tag do not match, raise an error
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
            # 'response_schema': TaggedSentences,   # Constrained decoding with pydantic schema (response_schema) caused segmentation problems
            'temperature': 0.0,
            # 'top_k': 1,
        },
    )

    if response.text is None:
        raise ValueError("No response from Gemini API. Please check your API key and network connection.")
    
    resp_dict = json.loads(response.text)
    res = TaggedSentences(**resp_dict)
    return res


# --- Example Usage ---
if __name__ == "__main__":
    UD_ENGLISH_TEST = './UD_English-EWT/en_ewt-ud-test.conllu'
    BATCH_SIZE = 5
    INPUT_FILE = 'hard_sentences.json'
    OUTPUT_FILE = 'hard_sentences_gemini.json'


    # Step 1: Load the test sentences
    # test_sentences, test_original = read_conllu(UD_ENGLISH_TEST)
    with open(INPUT_FILE, 'r') as f:
        input_data = [s for s in json.load(f) if 1 <= s['errors'] <= 3]
    test_sentences = [inp['tags'] for inp in input_data]
    test_original = [inp['original'] for inp in input_data]


    # Step 2: Perform POS tagging
    output = []

    sent_tok_err_count = 0
    batch_mismatch = 0

    total_tok_err_count = 0
    missing_tok_err_count = 0
    fixed_tok_err_count = 0

    for i in tqdm(range(0, len(test_original), BATCH_SIZE), desc="Tagging sentences", unit="batch"):
        # Step 2.1: Convert batch to TaggedSentences
        batch = test_original[i:i + BATCH_SIZE]
        batch_correct = TaggedSentences.from_2d_tuples(test_sentences[i:i + BATCH_SIZE])
        batch_correct_tokens = [[tags.token for tags in correct.sentence_tags] for correct in batch_correct.sentences]
        

        # Step 2.2: Tag the sentences using LLM
        batch_predicted = tag_sentences_ud(batch, batch_correct_tokens)


        # Step 2.3: Check for errors

        # Check if the number of sentences in the response matches the number of input sentences
        if len(batch) != len(batch_predicted.sentences):
            print(f"⚠️ Warning: The number of sentences in the response ({len(batch_predicted.sentences)}) does not match the number of input sentences ({len(batch)}) in batch {i}. Skipping this batch.")
            batch_mismatch += 1
            continue

        # Check LLM tokenization
        for sentence, correct, predicted in zip(batch, batch_correct.sentences, batch_predicted.sentences):
            correct_tags = [sentence_tag for sentence_tag in correct.sentence_tags]
            predicted_tags = [sentence_tag for sentence_tag in predicted.sentence_tags]

            correct_tokens = [ct.token for ct in correct_tags]
            predicted_tokens = [tt.token for tt in predicted_tags]

            if correct_tokens != predicted_tokens:

                sent_tok_err_count += 1

                # align the tokens
                aligned_predicted_tags = []
                matcher = SequenceMatcher(a=correct_tokens, b=predicted_tokens, autojunk=False)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        aligned_predicted_tags.extend(predicted_tags[j1:j2])
                    elif tag == "delete":
                        total_tok_err_count += i2 - i1
                        aligned_predicted_tags.extend([TokenPOS(token=ct.token, pos_tag=None) for ct in correct_tags[i1:i2]])
                        missing_tok_err_count += i2 - i1
                    elif tag == "replace":
                        total_tok_err_count += i2 - i1
                        for ct, pt in zip(correct_tags[i1:i2], predicted_tags[i1:i2]):
                            if pt.token.strip() == ct.token:
                                pt.token = pt.token.strip()
                                fixed_tok_err_count += 1
                            aligned_predicted_tags.append(pt)
                
                if [apt.token for apt in aligned_predicted_tags] != correct_tokens:
                    print(f"⚠️ Warning: Wrong tokenization. Trying to align predicted tokens to correct tokens.")
                    print(f"  Correct  : {correct_tokens}")
                    print(f"  Predicted: {predicted_tokens}")
                    print(f"  Aligned  : {[tag.token for tag in aligned_predicted_tags]}")
                
                predicted.sentence_tags = aligned_predicted_tags
            

            # Step 2.4: add correct and predicted tags to the output where needed
            output.append({
                "sentence": sentence,
                "tags": [
                    {
                        **({"predicted_token": pt.token}),
                        **({"correct_token": ct.token} if ct.token != pt.token else {}),
                        **({"predicted_tag": pt.pos_tag}),
                        **({"correct_tag": ct.pos_tag} if ct.pos_tag != pt.pos_tag else {}),
                    }
                    for ct, pt in zip(correct.sentence_tags, predicted.sentence_tags)
                ]
            })
    

    # Step 3: Save the output to a JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)


    # Step 4: Print summary
    if sent_tok_err_count > 0:
        print(f"❌ Error: {sent_tok_err_count} sentences had tokenization errors.")
        print(f"❌ Error: {total_tok_err_count} tokens had tokenization errors: ")
        print(f"          ✅ {total_tok_err_count - fixed_tok_err_count} of them were not fixed") if fixed_tok_err_count > 0 else None
        print(f"          ❌ {missing_tok_err_count} of the tokens are not provided by the LLM.")
    if batch_mismatch > 0:
        print(f"❌ Error: {batch_mismatch} batches had a batch size mismatch (the number of response sentences was different than the number of input sentences).")

    print(f"✅ Successfully tagged {len(test_original)} sentences.")
    print(f"✅ Successfully saved output to {OUTPUT_FILE}.")
    


# --- Testing original vs tokenized sentences ---

def untag(tagged_sentence):
    """Extract just the tokens from a tagged sentence"""
    return [token for token, _ in tagged_sentence]

def compare_original_vs_tokenized(test_data_path: str, sample_size: int = 50, batch_size: int = 5) -> dict:
    """
    Compare POS tagging performance between original and tokenized sentences
    
    Args:
        test_data_path: Path to the test data in CoNLL format
        sample_size: Number of sentences to sample for the test
        batch_size: Number of sentences to process in each API call
        
    Returns:
        Dictionary with comparison results
    """
    # Load test data
    test_sentences, test_original = read_conllu(test_data_path)
    
    # Limit to sample size
    test_sentences = test_sentences[:sample_size]
    test_original = test_original[:sample_size]
    
    results = {
        'original_accuracy': 0,
        'tokenized_accuracy': 0,
        'original_only_correct': 0,
        'tokenized_only_correct': 0,
        'both_correct': 0,
        'both_incorrect': 0,
        'original_tokenization_errors': 0,
        'tokenized_tokenization_errors': 0,
        'examples': []
    }
    
    # Prepare batched inputs
    original_batches = [test_original[i:i+batch_size] for i in range(0, len(test_original), batch_size)]
    
    tokenized_sentences = []
    for sentence in test_sentences:
        tokens = untag(sentence)
        tokenized_sentences.append(tokens)
    
    tokenized_batches = [tokenized_sentences[i:i+batch_size] for i in range(0, len(tokenized_sentences), batch_size)]
    
    # Process original sentences
    original_results = []
    for batch in tqdm(original_batches, desc="Processing original sentences"):
        try:
            batch_results = tag_sentences_ud(texts_to_tag=batch)
            original_results.append(batch_results)
        except Exception as e:
            print(f"Error processing original batch: {e}")
            # Add empty results to maintain order
            original_results.append(None)
    
    # Process tokenized sentences
    tokenized_results = []
    for batch in tqdm(tokenized_batches, desc="Processing tokenized sentences"):
        try:
            batch_results = tag_sentences_ud(tokens_to_tag=batch)
            tokenized_results.append(batch_results)
        except Exception as e:
            print(f"Error processing tokenized batch: {e}")
            # Add empty results to maintain order
            tokenized_results.append(None)
    
    # Flatten results
    flat_original_results = []
    for batch_result in original_results:
        if batch_result is not None:
            flat_original_results.extend(batch_result.sentences)
        else:
            # Add None placeholders to maintain alignment
            flat_original_results.extend([None] * batch_size)
    
    flat_tokenized_results = []
    for batch_result in tokenized_results:
        if batch_result is not None:
            flat_tokenized_results.extend(batch_result.sentences)
        else:
            # Add None placeholders to maintain alignment
            flat_tokenized_results.extend([None] * batch_size)
    
    # Trim to actual sample size
    flat_original_results = flat_original_results[:sample_size]
    flat_tokenized_results = flat_tokenized_results[:sample_size]
    
    # Evaluate and compare
    total_tokens = 0
    total_original_correct = 0
    total_tokenized_correct = 0
    
    for i, (gold_sentence, original_result, tokenized_result) in enumerate(zip(
        test_sentences, flat_original_results, flat_tokenized_results)):
        
        # Skip if either result is None
        if original_result is None or tokenized_result is None:
            continue
        
        gold_tokens = untag(gold_sentence)
        gold_tags = [tag for _, tag in gold_sentence]
        
        # Check original result
        original_tokens = [tag.token for tag in original_result.sentence_tags]
        original_tags = [tag.pos_tag for tag in original_result.sentence_tags]
        
        if len(original_tokens) != len(gold_tokens):
            results['original_tokenization_errors'] += 1
            # Try to align tokens for fair comparison
            aligned_original_tags = align_tokens(gold_tokens, original_tokens, original_tags)
        else:
            aligned_original_tags = original_tags
        
        # Check tokenized result
        tokenized_tokens = [tag.token for tag in tokenized_result.sentence_tags]
        tokenized_tags = [tag.pos_tag for tag in tokenized_result.sentence_tags]
        
        if len(tokenized_tokens) != len(gold_tokens):
            results['tokenized_tokenization_errors'] += 1
            # Align tokens for fair comparison
            aligned_tokenized_tags = align_tokens(gold_tokens, tokenized_tokens, tokenized_tags)
        else:
            aligned_tokenized_tags = tokenized_tags
        
        # Count correct tags
        original_correct = sum(p == g for p, g in zip(aligned_original_tags, gold_tags))
        tokenized_correct = sum(p == g for p, g in zip(aligned_tokenized_tags, gold_tags))
        
        # Update counters
        total_tokens += len(gold_tokens)
        total_original_correct += original_correct
        total_tokenized_correct += tokenized_correct
        
        # Check sentence-level correctness
        original_perfect = original_correct == len(gold_tokens)
        tokenized_perfect = tokenized_correct == len(gold_tokens)
        
        if original_perfect and tokenized_perfect:
            results['both_correct'] += 1
        elif original_perfect and not tokenized_perfect:
            results['original_only_correct'] += 1
        elif not original_perfect and tokenized_perfect:
            results['tokenized_only_correct'] += 1
        else:
            results['both_incorrect'] += 1
        
        # Save interesting examples
        if original_correct != tokenized_correct:
            results['examples'].append({
                'sentence_id': i,
                'original_text': test_original[i],
                'tokenized_text': ' '.join(gold_tokens),
                'gold_tokens': gold_tokens,
                'gold_tags': gold_tags,
                'original_tokens': original_tokens,
                'original_tags': aligned_original_tags,
                'tokenized_tokens': tokenized_tokens,
                'tokenized_tags': aligned_tokenized_tags,
                'original_correct': original_correct,
                'tokenized_correct': tokenized_correct,
                'better': 'original' if original_correct > tokenized_correct else 'tokenized'
            })
    
    # Calculate overall accuracy
    results['original_accuracy'] = total_original_correct / total_tokens if total_tokens > 0 else 0
    results['tokenized_accuracy'] = total_tokenized_correct / total_tokens if total_tokens > 0 else 0
    
    return results

def align_tokens(gold_tokens, pred_tokens, pred_tags):
    """Align predicted tags with gold tokens when tokenization differs"""
    from difflib import SequenceMatcher
    
    matcher = SequenceMatcher(a=gold_tokens, b=pred_tokens)
    aligned_tags = ['UNK'] * len(gold_tokens)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if i < len(aligned_tags) and j < len(pred_tags):
                    aligned_tags[i] = pred_tags[j]
        elif tag == 'replace' or tag == 'delete':
            # Use nearest available tag when possible
            if j1 < len(pred_tags):
                for i in range(i1, i2):
                    if i < len(aligned_tags):
                        aligned_tags[i] = pred_tags[min(j1, len(pred_tags)-1)]
    
    return aligned_tags

# New command for original vs tokenized comparison
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "compare-formats":
    UD_ENGLISH_TEST = './UD_English-EWT/en_ewt-ud-test.conllu'
    SAMPLE_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    OUTPUT_FILE = 'format_comparison_results.json'
    
    print(f"Comparing original vs tokenized format using {SAMPLE_SIZE} samples with batch size {BATCH_SIZE}")
    results = compare_original_vs_tokenized(UD_ENGLISH_TEST, SAMPLE_SIZE, BATCH_SIZE)
    
    # Print summary
    print("\n=== Results Summary ===")
    print(f"Original sentence accuracy: {results['original_accuracy']:.4f}")
    print(f"Tokenized sentence accuracy: {results['tokenized_accuracy']:.4f}")
    print(f"Sentences where only original was perfect: {results['original_only_correct']}")
    print(f"Sentences where only tokenized was perfect: {results['tokenized_only_correct']}")
    print(f"Sentences where both were perfect: {results['both_correct']}")
    print(f"Sentences where both had errors: {results['both_incorrect']}")
    print(f"Original tokenization errors: {results['original_tokenization_errors']}")
    print(f"Tokenized tokenization errors: {results['tokenized_tokenization_errors']}")
    
    # Save detailed results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {OUTPUT_FILE}")
    
    # Show examples where formats differ
    print("\n=== Interesting Examples ===")
    for i, example in enumerate(results['examples']):
        if i >= 5:  # Limit to first 5 examples
            print(f"... and {len(results['examples']) - 5} more examples")
            break
            
        print(f"\nExample {i+1} (better: {example['better']}):")
        print(f"Original text: {example['original_text']}")
        print(f"Tokenized text: {example['tokenized_text']}")
        print(f"Original correct: {example['original_correct']}/{len(example['gold_tokens'])}")
        print(f"Tokenized correct: {example['tokenized_correct']}/{len(example['gold_tokens'])}")
        
        # Show mismatched tags
        print("Tag differences:")
        for j, (gold_token, gold_tag, orig_tag, tok_tag) in enumerate(zip(
            example['gold_tokens'], example['gold_tags'], 
            example['original_tags'], example['tokenized_tags'])):
            
            if orig_tag != tok_tag:
                print(f"  '{gold_token}': gold={gold_tag}, original={orig_tag}, tokenized={tok_tag}")
