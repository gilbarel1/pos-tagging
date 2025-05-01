# --- Imports ---
import os
import sys
from enum import Enum
from typing import List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from pos_defs import ALL_DEFS


load_dotenv()

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
    pos_tag: UDPosTag = Field(description="The part-of-speech tag for the token.")

class SentencePOS(BaseModel):
    """Represents a sentence with its tokens and their POS tags."""
    sentence_tags: List[TokenPOS] = Field(description="A list of tokens with their POS tags.")

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

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

def tag_sentences_ud(texts_to_tag: List[str]) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    # Construct the prompt
    prompt = f"""# Parts-of-Speech

Below is a list of all 17 parts-of-speech along with their definitions.

{"\n\n\n".join(ALL_DEFS)}


# Your Task

You are given a list of sentences. You should identify the parts-of-speech in the sentence and classify them.

## Sentences

{"\n\n".join(texts_to_tag)}
"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TaggedSentences,
        },
    )
    # Use the response as a JSON string.
    print(response.text)

    # Use instantiated objects.
    res: TaggedSentences = response.parsed
    return res


# --- Example Usage ---
if __name__ == "__main__":
    # example_text = "The quick brown fox jumps over the lazy dog."
    example_texts = ["What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?"]
    # example_text = "Google Search is a web search engine developed by Google LLC."
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    print(f"\nTagging texts: \"{example_texts}\"")

    tagged_result = tag_sentences_ud(example_texts)

    if tagged_result:
        print("\n--- Tagging Results ---")
        for s in tagged_result.sentences:
            for sentence_tag in s.sentence_tags:
                token = sentence_tag.token
                tag = sentence_tag.pos_tag
                # Handle potential None for pos_tag if model couldn't assign one
                ctag = tag if tag is not None else "UNKNOWN"
                print(f"Token: {token:<15} {str(ctag)}")
                print("----------------------")
    else:
        print("\nFailed to get POS tagging results.")