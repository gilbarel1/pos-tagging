# --- Imports ---
import os
import sys
import json
from enum import Enum
from typing import List, Union, Optional, Tuple, Self
from difflib import SequenceMatcher

import nltk
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tqdm import tqdm
from ratelimiter import RateLimiter

from pos_defs import ALL_DEFS, SHORT_DEFS
from utils import read_conllu


load_dotenv()
nltk.download('punkt_tab')

# 
gemini_model = 'gemini-2.0-flash-lite'


# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    ADJ = "ADJ"  # Adjective
    ADP = "ADP"  # Adposition
    ADV = "ADV"  # Adverb
    AUX = "AUX"  # Auxiliary verb
    CCONJ = "CCONJ"  # Coordinating conjunction
    DET = "DET"  # Determiner
    INTJ = "INTJ"  # Interjection
    NOUN = "NOUN"  # Noun
    NUM = "NUM"  # Numeral
    PART = "PART"  # Particle
    PRON = "PRON"  # Pronoun
    PROPN = "PROPN"  # Proper noun
    PUNCT = "PUNCT"  # Punctuation
    SCONJ = "SCONJ"  # Subordinating conjunction
    SYM = "SYM"  # Symbol
    VERB = "VERB"  # Verb
    X = "X"  # Other


# --- Define Pydantic Models for Structured Output ---
class TokenPOS(BaseModel):
    """Represents a token with its part-of-speech (POS) tag."""
    token: str = Field(description="The token itself.")
    pos_tag: Optional[UDPosTag] = Field(description="The part-of-speech tag for the token. None if not tagged.")


    @classmethod
    def from_tuple(cls, tup: Tuple[str, str]) -> Self:
        """
        Create a TokenPOS instance from a tuple of (token, pos_tag).

        Args:
            tup: A tuple containing the token and its POS tag.

        Returns:
            An instance of TokenPOS.
        """
        token, pos_tag = tup
        return cls(token=token, pos_tag=UDPosTag(pos_tag))

class SentencePOS(BaseModel):
    """Represents a sentence with its tokens and their POS tags."""
    sentence_tags: List[TokenPOS] = Field(description="A list of tokens with their POS tags.")

    @classmethod
    def from_tuples(cls, tuples: List[Tuple[str, str]]) -> Self:
        """
        Create a SentencePOS instance from a list of tuples.

        Args:
            tuples: A list of tuples containing tokens and their POS tags.

        Returns:
            An instance of SentencePOS.
        """
        sentence_tags = [TokenPOS.from_tuple(tup) for tup in tuples]
        return cls(sentence_tags=sentence_tags)

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

    @classmethod
    def from_2d_tuples(cls, tuples_matrix: List[List[Tuple[str, str]]]) -> Self:
        """
        Create a TaggedSentences instance from a list of lists of tuples.

        Args:
            tuples: A list of lists of tuples containing tokens and their POS tags.

        Returns:
            An instance of TaggedSentences.
        """
        sentences = [SentencePOS.from_tuples(tuples) for tuples in tuples_matrix]
        return cls(sentences=sentences)

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
@RateLimiter(max_calls=15, period=60)
def tag_sentences_ud(
        texts_to_tag: Union[str, List[str]],
        tokens_to_tag: Optional[Union[List[str], List[List[str]]]] = None
    ) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
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
    prompt = f"""# Universal POS tags

Below is a list of 17 tags. These tags mark the core part-of-speech categories.

| Open class words | Closed class words | Other |
|------------------|--------------------|-------|
| ADJ, ADV, INTJ, NOUN, PROPN, VERB | ADP, AUX, CCONJ, DET, NUM, PART, PRON, SCONJ | PUNCH, SYM, X |

Alphabetical listing:

{"\n".join(map(lambda x: f"- {x}", SHORT_DEFS))}


# Your Task

You are a POS tagger. You are given a list of sentences and their corresponding list of tokens below.
The list of tokens is in the format of a comma separated list (for example: ['tok1', 'tok2', 'tok3']) where each token is surrounded by apostrophes (or double quotes if the token contains an apostrophe).

{"\n\n".join(map(lambda txt_tokens: f"Sentence: {txt_tokens[0]}\nTokens: {txt_tokens[1]}", zip(texts_to_tag, tokens_to_tag)))}

You MUST classify all tokens in every sentence to its part-of-speech tag.
In your response, the tokens should be EXACTLY the input tokens, **DO NOT change the given tokenization**. You MUST specify a tag for each token."""

    # Send prompt to the Gemini API
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TaggedSentences,
            'temperature': 0.0,
            'top_k': 1,
        },
    )

    # Use instantiated objects.
    res: TaggedSentences = response.parsed  # type: ignore
    return res


# --- Example Usage ---
if __name__ == "__main__":
    UD_ENGLISH_TEST = './UD_English-EWT/en_ewt-ud-test.conllu'
    BATCH_SIZE = 5
    INPUT_FILE = 'hard_sentences.json'
    OUTPUT_FILE = 'hard_sentences_gemini.json'

    # test_sentences, test_original = read_conllu(UD_ENGLISH_TEST)
    with open(INPUT_FILE, 'r') as f:
        input_data = json.load(f)
    
    test_sentences = [inp['tags'] for inp in input_data]
    test_original = [inp['original'] for inp in input_data]

    output = []

    tok_err_count = 0

    for i in tqdm(range(0, len(test_original), BATCH_SIZE), desc="Tagging sentences", unit="batch"):
        batch = test_original[i:i + BATCH_SIZE]
        batch_correct = TaggedSentences.from_2d_tuples(test_sentences[i:i + BATCH_SIZE])
        batch_correct_tokens = [[tags.token for tags in correct.sentence_tags] for correct in batch_correct.sentences]
        
        batch_predicted = None
        while batch_predicted is None or len(batch) != len(batch_predicted.sentences):
            batch_predicted = tag_sentences_ud(batch, batch_correct_tokens)

        for sentence, correct, predicted in zip(batch, batch_correct.sentences, batch_predicted.sentences):
            # check that every word in correct is in predicted
            correct_tags = [sentence_tag for sentence_tag in correct.sentence_tags]
            predicted_tags = [sentence_tag for sentence_tag in predicted.sentence_tags]

            correct_tokens = [ct.token for ct in correct_tags]
            predicted_tokens = [tt.token for tt in predicted_tags]

            if correct_tokens != predicted_tokens:
                print(f"⚠️ Warning: Wrong tokenization. Alignning predicted tokens to correct tokens (with POS None for missing tokens).")

                tok_err_count += 1

                # align the tokens
                aligned_predicted_tags = []
                matcher = SequenceMatcher(a=correct_tokens, b=predicted_tokens, autojunk=False)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        aligned_predicted_tags.extend(predicted_tags[i1:i2])
                    elif tag == "delete":
                        aligned_predicted_tags.extend([TokenPOS(token=ct.token, pos_tag=None) for ct in correct_tags[i1:i2]])
                    elif tag == "replace":
                        for ct, pt in zip(correct_tags[i1:i2], predicted_tags[i1:i2]):
                            if pt.token.strip() == ct.token:
                                pt.token = pt.token.strip()
                            aligned_predicted_tags.append(pt)
                        
                print(f"  Correct  : {correct_tokens}")
                print(f"  predicted: {predicted_tokens}")
                print(f"  Aligned  : {[tag.token for tag in aligned_predicted_tags]}")
                
                predicted.sentence_tags = aligned_predicted_tags
            
            output.append({
                "sentence": sentence,
                "tags": [
                    {
                        **({"correct_token": ct.token} if ct.token != pt.token else {}),
                        **({"predicted_token": pt.token}),
                        **({"correct_tag": ct.pos_tag} if ct.pos_tag != pt.pos_tag else {}),
                        **({"predicted_tag": pt.pos_tag}),
                    }
                    for ct, pt in zip(correct.sentence_tags, predicted.sentence_tags)
                ]
            })
    
    # Save the output to a JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    if tok_err_count > 0:
        print(f"⚠️ Warning: {tok_err_count} sentences had tokenization errors.")
    print(f"✅ Successfully tagged {len(test_original)} sentences.")
    print(f"✅ Successfully saved output to {OUTPUT_FILE}.")
