# radlex_reconstruction.py
import pandas as pd
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=False)
nltk.download('omw-1.4', quiet=False)
nltk.download('averaged_perceptron_tagger', quiet=False)
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

def get_lemma(word):
    doc = nlp(word)
    if doc:
        token = doc[0]
        pos = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}.get(token.pos_, 'n')
        return lemmatizer.lemmatize(word, pos)
    return lemmatizer.lemmatize(word)

dictionary_csv = pd.read_excel(r"Radlex.xls")
prefer_name = dictionary_csv['Preferred Label'].dropna().tolist()
prefer_synonym = dictionary_csv['Synonyms'].dropna().tolist()

radlex_lemmas = set()
for name in prefer_name:
    try:
        words = name.split()
        print(words)  ## For Debug usage
        for w in words:
            radlex_lemmas.add(get_lemma(w.lower()))
    except:
        print("Error Detect, Please Check")
        print(name)

for syn in prefer_synonym:
    try:
        syns = syn.split('|')  # Assume | separator; change to ',' if needed
        for s in syns:
            words = s.strip().split()
            print(words) ## For Debug usage
            for w in words:
                radlex_lemmas.add(get_lemma(w.lower()))
    except:
        print("Error Detect, Please Check")
        print(name)

case_final = pd.DataFrame({'lemma': list(radlex_lemmas)})
case_final.to_csv(r"radlex_lemmas.csv", index=False)
print("RadLex lemmas saved to radlex_lemmas.csv")