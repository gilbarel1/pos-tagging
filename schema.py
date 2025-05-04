from enum import Enum
from typing import Optional, List, Tuple, Self

from pydantic import BaseModel, Field


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
    

class TaggerErrorExplanation(BaseModel):
    token: str = Field(description="The token that was tagged incorrectly. MUST be exactly the same as the input token.")
    predicted_tag: UDPosTag = Field(description="The predicted part-of-speech tag.")
    correct_tag: UDPosTag = Field(description="The correct part-of-speech tag.")
    explanation: str = Field(description="A possible explanation of why the tagger predicted the wrong label instead of the correct label.")
    category: str = Field(description="Category of the error.")