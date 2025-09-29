# keyword_extraction.py
import argparse
import pandas as pd
from keybert import KeyBERT
import spacy
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

def get_lemma(word):
    doc = nlp(word)
    if doc:
        token = doc[0]
        pos = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}.get(token.pos_, 'n')
        return lemmatizer.lemmatize(word, pos)
    return lemmatizer.lemmatize(word)

radlex_df = pd.read_csv(r"Automatic Keyword Adaption Session/extract_Dictionary.csv")
radlex_lemmas = set(radlex_df['Preferred Label'].tolist()).union(set(radlex_df['Synonyms'].tolist()))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['iuxray', 'mimic_cxr']) ## Redundant Command
parser.add_argument('--split', required=True, choices=['train', 'test', 'val']) ## Redundant Command
parser.add_argument('--input_csv', required=True)
parser.add_argument('--output_csv', required=True)
args = parser.parse_args()

kw_model = KeyBERT()
report_csv = pd.read_csv(args.input_csv)
case_number = report_csv['Case_num'].tolist() if 'Case_num' in report_csv.columns else report_csv['Image_id'].tolist()
report_list = report_csv['Ground-Truth Report'].tolist() if 'Ground-Truth Report' in report_csv.columns else report_csv['Ground-Truth'].tolist()

predicted_keyword_list = []
predicted_keyword_list_after_filter_l1 = []

for s1 in range(len(report_list)):
    predicted_keyword_list.append([])
    predicted_keyword_list_after_filter_l1.append([])
    doc = report_list[s1]
    if pd.isna(doc):
        continue
    doc_list = doc.split(".")[:-1]
    kw_model = KeyBERT()
    keywords_1 = []
    for i in range(len(doc_list)):
        keyword_1_list = kw_model.extract_keywords(doc_list[i], keyphrase_ngram_range=(1, 1), stop_words=None)
        print(keyword_1_list)  ## For Debug usage
        for j in range(len(keyword_1_list)):
            kw = keyword_1_list[j][0].lower()
            predicted_keyword_list[s1].append(kw)
            lemma = get_lemma(kw)
            if lemma in radlex_lemmas:
                predicted_keyword_list_after_filter_l1[s1].append(lemma)
    predicted_keyword_list[s1] = list(set(predicted_keyword_list[s1]))
    predicted_keyword_list_after_filter_l1[s1] = list(set(predicted_keyword_list_after_filter_l1[s1]))

case_final = pd.DataFrame(columns=['Case_num', 'Ground-Truth', "Total_keyword", "Level1_keywords"],
                          data=list(zip(case_number, report_list, predicted_keyword_list,
                                        predicted_keyword_list_after_filter_l1)))
case_final.to_csv(args.output_csv, index=False)
print(f"Filtered keywords saved to {args.output_csv}")