import json
from typing import Optional, List, Union

from pos_defs import SHORT_DEFS
from schema import TaggedSentences

def tagger_prompt(tokens_to_tag: Union[str, List[str]]) -> str:
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
