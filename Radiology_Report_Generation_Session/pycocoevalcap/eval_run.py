import json
import pandas as pd

from metrics import compute_scores
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

total_csv=pd.read_csv(r"D:\Second_Stage_PhD_Project\PhD_Stage2\Result\total_result.csv")
print(total_csv.columns)

print('tokenization...')
tokenizer = PTBTokenizer()

our_json=[]
r2gen_json=[]
distgen2_json=[]
gt_json=[]

"""
for i in range(len(total_csv)):
    gt_json[total_csv['Case ID'][i]] = total_csv['GT'][i]
    our_json[total_csv['Case ID'][i]]=total_csv['Ours'][i]
    r2gen_json[total_csv['Case ID'][i]] = total_csv['R2Gen'][i]
    distgen2_json[total_csv['Case ID'][i]] = total_csv['DistGen2'][i]
"""
#{"image_id": 404464, "caption": "
gt_json.append({"image_id":total_csv['Case ID'][15],"caption":total_csv['GT'][15]})
our_json.append({"image_id":total_csv['Case ID'][15],"caption":total_csv['Ours'][15]})
r2gen_json.append({"image_id":total_csv['Case ID'][15],"caption":total_csv['R2Gen'][15]})
distgen2_json.append({"image_id":total_csv['Case ID'][15],"caption":total_csv['DistGen2'][15]})

print(gt_json)


gt_json=tokenizer.tokenize(gt_json)
r2gen_json=tokenizer.tokenize(r2gen_json)

print(compute_scores(gt_json,r2gen_json))

