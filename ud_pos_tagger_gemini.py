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

from utils import read_conllu, untag
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
    INPUT_FILE = './datasets/hard_sentences.json'
    OUTPUT_FILE = './datasets/hard_sentences_gemini.json'


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

def test_tokenization_impact():
    """
    Test the impact of using original vs tokenized sentences on the LLM tagger's performance.
    Outputs two JSON files for comparison.
    """
    UD_ENGLISH_TEST = '../UD_English-EWT/en_ewt-ud-test.conllu'
    OUTPUT_FILE_ORIGINAL = './datasets/original_sentences_results.json'
    OUTPUT_FILE_TOKENIZED = './datasets/tokenized_sentences_results.json'
    BATCH_SIZE = 5
    MAX_SENTENCES = 200  # Adjust this to test on more or fewer sentences
    
    # Step 1: Load the test sentences
    print("Loading test sentences...")
    test_sentences, test_original = read_conllu(UD_ENGLISH_TEST)
    
    # Limit the number of sentences for testing
    test_sentences = test_sentences[:MAX_SENTENCES]
    test_original = test_original[:MAX_SENTENCES]
    
    # Step 2: Create tokenized versions
    print("Creating tokenized versions...")
    tokenized_sentences = []
    for sentence in test_sentences:
        tokens = untag(sentence)
        tokenized_text = " ".join(tokens)
        tokenized_sentences.append(tokenized_text)
    
    # Step 3: Process original sentences
    print("Processing original sentences...")
    original_results = process_sentences(test_original, test_sentences, BATCH_SIZE)
    
    # Step 4: Process tokenized sentences
    print("Processing tokenized sentences...")
    tokenized_results = process_sentences(tokenized_sentences, test_sentences, BATCH_SIZE)
    
    # Step 5: Save results
    print(f"Saving results to {OUTPUT_FILE_ORIGINAL} and {OUTPUT_FILE_TOKENIZED}")
    with open(OUTPUT_FILE_ORIGINAL, 'w') as f:
        json.dump(original_results, f, indent=2)
    
    with open(OUTPUT_FILE_TOKENIZED, 'w') as f:
        json.dump(tokenized_results, f, indent=2)
    
    print("Done!")

def process_sentences(input_sentences, reference_sentences, batch_size):
    """
    Process sentences in batches and return results in the expected format.
    
    Args:
        input_sentences: List of sentences to be tagged (either original or tokenized)
        reference_sentences: Original sentences with correct tags for reference
        batch_size: Number of sentences to process in each batch
    
    Returns:
        List of dictionaries with tagged results
    """
    output = []
    sent_tok_err_count = 0
    batch_mismatch = 0
    total_tok_err_count = 0
    missing_tok_err_count = 0
    fixed_tok_err_count = 0

    for i in tqdm(range(0, len(input_sentences), batch_size), desc="Tagging sentences", unit="batch"):
        # Step 1: Prepare the batch
        batch = input_sentences[i:i + batch_size]
        batch_correct = TaggedSentences.from_2d_tuples(reference_sentences[i:i + batch_size])
        batch_correct_tokens = [[tags.token for tags in correct.sentence_tags] for correct in batch_correct.sentences]
        
        # Step 2: Tag the sentences using LLM
        try:
            batch_predicted = tag_sentences_ud(batch, batch_correct_tokens)
        except Exception as e:
            print(f"Error processing batch {i}: {str(e)}")
            continue
            
        # Step 3: Check if the number of sentences in the response matches the number of input sentences
        if len(batch) != len(batch_predicted.sentences):
            print(f"⚠️ Warning: The number of sentences in the response ({len(batch_predicted.sentences)}) does not match the number of input sentences ({len(batch)}) in batch {i}. Skipping this batch.")
            batch_mismatch += 1
            continue

        # Step 4: Process and align results
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

            # Step 5: Add correct and predicted tags to the output
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

    # Print summary
    if sent_tok_err_count > 0:
        print(f"❌ Error: {sent_tok_err_count} sentences had tokenization errors.")
        print(f"❌ Error: {total_tok_err_count} tokens had tokenization errors: ")
        print(f"          ✅ {total_tok_err_count - fixed_tok_err_count} of them were not fixed") if fixed_tok_err_count > 0 else None
        print(f"          ❌ {missing_tok_err_count} of the tokens are not provided by the LLM.")
    if batch_mismatch > 0:
        print(f"❌ Error: {batch_mismatch} batches had a batch size mismatch (the number of response sentences was different than the number of input sentences).")

    return output


def tag_segmented_failed_sentences():
    """
    Tags the segmented failed sentences and saves the results to a new JSON file.
    Uses the segmented tokens from failed_sentences_segmented.json as input.
    """
    INPUT_FILE = './datasets/failed_sentences_segmented.json'
    OUTPUT_FILE = './datasets/failed_sentences_tagged.json'
    
    # Step 1: Load the segmented sentences
    print(f"Loading segmented sentences from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            segmented_sentences = json.load(f)
        
        # Check if we have valid data
        if not isinstance(segmented_sentences, list) or not segmented_sentences:
            raise ValueError(f"Invalid or empty data in {INPUT_FILE}")
            
        print(f"Successfully loaded {len(segmented_sentences)} segmented sentences")
    except Exception as e:
        print(f"Error loading segmented sentences: {str(e)}")
        return
    
    # Step 2: Prepare the data for tagging
    # Extract original sentences and predicted tokens
    original_sentences = [entry["original"] for entry in segmented_sentences]
    tokenized_sentences = [" ".join(entry["predicted_tokens"]) for entry in segmented_sentences]
    tokens_list = [entry["predicted_tokens"] for entry in segmented_sentences]
    
    # Step 3: Tag the sentences using the LLM
    results = []
    batch_size = 1  # Process one sentence at a time due to complexity
    
    print(f"Tagging {len(segmented_sentences)} sentences...")
    for i in tqdm(range(len(segmented_sentences)), desc="Tagging sentences", unit="sentence"):
        try:
            # Get original and tokenized data for this sentence
            original = original_sentences[i]
            tokens = tokens_list[i]
            tokenized = tokenized_sentences[i]
            
            # Tag the tokens using LLM
            tagged_result = tag_sentences_ud([tokenized], [tokens])
            
            # Extract the tags
            if tagged_result.sentences and tagged_result.sentences[0].sentence_tags:
                pos_tags = [
                    {
                        "token": tag.token,
                        "pos_tag": tag.pos_tag
                    } for tag in tagged_result.sentences[0].sentence_tags
                ]
                
                # Create the result entry
                result_entry = {
                    "original": original,
                    "tokens": tokens,
                    "tags": pos_tags
                }
                results.append(result_entry)
            else:
                print(f"⚠️ Warning: No tags returned for sentence {i+1}: {original[:30]}...")
                results.append({
                    "original": original,
                    "tokens": tokens,
                    "error": "No tags returned by the LLM"
                })
                
        except Exception as e:
            print(f"Error processing sentence {i+1}: {str(e)}")
            results.append({
                "original": original_sentences[i],
                "tokens": tokens_list[i],
                "error": str(e)
            })
    
    # Step 4: Save the results
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Successfully tagged {len(results)} sentences.")
    print(f"Results saved to {OUTPUT_FILE}")

# if __name__ == "__main__":
    # Uncomment the line below to run the tokenization impact test
    # test_tokenization_impact()
    # tag_segmented_failed_sentences()