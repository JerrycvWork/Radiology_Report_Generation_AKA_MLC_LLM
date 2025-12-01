from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.spice.spice import Spice
import numpy as np

class Evaluator:
    def __init__(self) -> None:
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        self.evaluation_report = {}
        self.List_of_Score={}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)

        # From this point, some variables are named as in the original code
        # I have no idea why they name like these
        # The original code: https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
                    self.List_of_Score[m]=np.std(scs)

            else:
                self.evaluation_report[method] = score
                self.List_of_Score[method] = np.std(scores)

import pandas as pd
total_csv=pd.read_csv(r"D:\Second_Stage_PhD_Project\Tag_Generation\20240408\iuxray\Result_test_base.csv")


## IUX-ray Special

import pandas as pd
import xml.etree.ElementTree as ET
import glob
import os

import json
import shutil
import pandas as pd
import numpy as np

with open(r"D:\Second_Stage_PhD_Project\IUXray_Result\\annotation.json") as f:
    total_json=json.load(f)

print(total_json.keys())

print(total_json["test"])

test_id=[]

test_report=[]


for i in range(len(total_json["test"])):
    #dict_keys(['id', 'report', 'image_path', 'split'])
    #print(total_json["test"][i].keys())
    test_id.append(total_json["test"][i]['id'].split("_")[0].split("CXR")[-1])

print(test_id)

for i in range(len(total_csv)):
    if str(total_csv["Case_num"][i]) not in test_id:
        total_csv=total_csv.drop(i)



golden_reference = []
golden_reference=total_csv['Ground-Truth']

golden_reference = {k: [{'caption': v}] for k, v in enumerate(golden_reference)}

candidate_reference = []
#candidate_reference=total_csv['Ours']
#candidate_reference=total_csv['R2Gen']
candidate_reference=total_csv['Result']

candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}

evaluator = Evaluator()

evaluator.do_the_thing(golden_reference, candidate_reference)

print(evaluator.evaluation_report)
print(evaluator.List_of_Score)

report_copy=evaluator.evaluation_report

print(list(report_copy.keys()))


