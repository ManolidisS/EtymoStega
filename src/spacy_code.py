from src.config import VERB_POS
import spacy
spacy.require_cpu()
nlp = spacy.load("en_core_web_lg")

def exact_first_verb(text:str) -> str | None:
    doc = nlp(text)
    for token in doc:
        if token.pos_ in VERB_POS:
            return token.text
    return None

def exact_verbs(text:str) -> list:
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ in VERB_POS]

def lemmatize(word:str) -> str:
    return nlp(word)[0].lemma_

def lemma_first_verb(text:str) -> str | None:
    first_verb = exact_first_verb(text)
    if first_verb:
        return lemmatize(first_verb)
    return None

def lemma_verbs(text:str) -> list:
    verbs = exact_verbs(text)
    verbs = [lemmatize(verb) for verb in verbs]
    return verbs