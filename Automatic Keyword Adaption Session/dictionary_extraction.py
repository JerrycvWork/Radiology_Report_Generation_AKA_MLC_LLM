import pandas as pd
from keybert import KeyBERT
## Load the Radiology Report

dictionary_csv=pd.read_excel(r"Automatic Keyword Adaption Session/Radlex.xls")

print(dictionary_csv.columns)

prefer_name=list(dictionary_csv['Preferred Label'])
prefer_synonym=list(dictionary_csv['Synonyms'])
prefer_definition=list(dictionary_csv['Definitions'])


print(len(prefer_name))
print(len(prefer_synonym))
print(len(prefer_definition))

case_final = pd.DataFrame(columns=['Preferred Label','Synonyms','Definitions'], data=list(zip(prefer_name,prefer_synonym,prefer_definition)))
case_final.to_csv(r"Automatic Keyword Adaption Session/extract_Dictionary.csv",index=False)



