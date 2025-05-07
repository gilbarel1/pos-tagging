import json
from typing import Optional, List, Union

from pos_defs import SHORT_DEFS, ALL_DEFS
from schema import TaggedSentences, UDPosTag, TaggerErrorExplanation

def tagger_prompt(tokens_to_tag: List[str]) -> str:
    return f"""# Universal POS tags

Below is a list of 17 tags. These tags mark the core part-of-speech categories.

| Open class words | Closed class words | Other |
|------------------|--------------------|-------|
| ADJ, ADV, INTJ, NOUN, PROPN, VERB | ADP, AUX, CCONJ, DET, NUM, PART, PRON, SCONJ | PUNCH, SYM, X |

Alphabetical listing:

{"\n".join(map(lambda x: f"- {x}", SHORT_DEFS))}


# Your Task

You are a POS tagger. You are given a list of pre-segmented sentences below. For each token in the sentence, you need to provide its part-of-speech tag. Tokens are separated by spaces.
Everything that follows the words "Segmented sentence: " in the same line is a sentence that should be tagged (even if its very short, long, or even empty).

{"\n\n".join(map(lambda sentence_tokens: f"Segmented sentence: {' '.join(sentence_tokens)}", tokens_to_tag))}

You MUST classify every token in every sentence to its part-of-speech tag.
In your response, the tokens should be EXACTLY the input tokens, **DO NOT change the given segmentation**. You MUST specify a tag for each token.

The output should be in a JSON that corresponds to the following schema:
```json
{json.dumps(TaggedSentences.model_json_schema(), indent=2)}
```"""


def explain_error_prompt(sentence_tokens: List[str], predicted_tags: List[UDPosTag], correct_tags: List[UDPosTag]) -> str:
    predictions_table = "| Token | Predicted tag | Correct tag | Error |\n" \
                        "|-------|---------------|-------------| ----- |\n"
    predictions_table += '\n'.join(
        [
            f"| {token} | {pred.value} | {correct.value} | {'YES' if pred != correct else 'NO'} |" 
            for token, pred, correct in zip(sentence_tokens, predicted_tags, correct_tags)
        ]
    )

    return f"""# Universal POS tags

Below is a list of 17 tags. These tags mark the core part-of-speech categories.

| Open class words | Closed class words | Other |
|------------------|--------------------|-------|
| ADJ, ADV, INTJ, NOUN, PROPN, VERB | ADP, AUX, CCONJ, DET, NUM, PART, PRON, SCONJ | PUNCH, SYM, X |

Alphabetical listing:

{"\n\n\n".join(map(lambda x: f"- {x}", ALL_DEFS))}

# Your Task

You are a POS tagger evaluator. You are given a segmented sentence (each token is separated by spaces), where for each token you get the predicted tag and the correct tag:

{' '.join(sentence_tokens)}
Number of errors: {len([0 for pred, correct in zip(predicted_tags, correct_tags) if pred != correct])}

{predictions_table}

Your task is to explain the errors made by the tagger. For each error, you should provide explanation and category of the error.

The output should be in a **JSON list** such that each item corresponds to the following schema:
```json
{json.dumps(TaggerErrorExplanation.model_json_schema(), indent=2)}
```"""

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