from pyconll.load import load_from_file


def read_conllu(path):
    data = load_from_file(path)
    tagged_sentences=[]
    original_sentences=[]
    t=0
    for sentence in data:
        original_sentences.append(sentence.text)
        tagged_sentence=[]
        for token in sentence:
            if token.upos:
                t+=1
                tagged_sentence.append((token.form if token.form else '*None*', token.upos))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences, original_sentences

def untag(tagged_sentence):
    """Extract just the tokens from a tagged sentence"""
    return [token for token, _ in tagged_sentence]