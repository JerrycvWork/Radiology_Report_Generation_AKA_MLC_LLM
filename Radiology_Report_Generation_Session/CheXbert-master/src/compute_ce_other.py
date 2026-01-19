from pprint import pprint

import pandas as pd

from metrics import compute_mlc, compute_mlc_for_std
import numpy as np

def main():
    """
    res_path = "/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/cvt2_DistGen2_Reset_Target_result_labeled_reports.csv"
    gts_path = "/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/cvt2_DistGen2_Reset_Target_groundtruth_labeled_reports.csv"
    res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
    res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

    label_set = res_data.columns[1:].tolist()
    res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
    res_data[res_data == -1] = 0
    gts_data[gts_data == -1] = 0

    met_csv = pd.read_csv(res_path)

    f1_ma=[]
    f1_mi=[]
    pre_ma=[]
    pre_mi=[]
    rec_ma=[]
    rec_mi=[]

    for i in range(len(res_data)):
        metrics = compute_mlc(gts_data[i,:], res_data[i,:], label_set)
        pprint(metrics)

        f1_ma.append(metrics['F1_MACRO'])
        f1_mi.append(metrics['F1_MICRO'])
        pre_ma.append(metrics['PRECISION_MACRO'])
        pre_mi.append(metrics['PRECISION_MICRO'])
        rec_ma.append(metrics['RECALL_MACRO'])
        rec_mi.append(metrics['RECALL_MICRO'])

    met_csv['F1_MACRO']=f1_ma
    met_csv['F1_MICRO'] = f1_mi
    met_csv['PRECISION_MACRO'] = pre_ma
    met_csv['PRECISION_MICRO'] = pre_mi
    met_csv['RECALL_MACRO'] = rec_ma
    met_csv['RECALL_MICRO'] = rec_mi

    met_csv.to_csv("/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/cvt2_DistGen2_Reset_Target_result_metrics.csv",index=False)


    metrics = compute_mlc(res_data, res_data, label_set)
    pprint(metrics)

    metrics = compute_mlc(gts_data, gts_data, label_set)
    pprint(metrics)
    """

    res_path = "/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/r2gen_result_revise_mimic_result_labeled_reports.csv"
    gts_path = "/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/r2gen_result_revise_mimic_groundtruth_labeled_reports.csv"
    res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
    res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

    label_set = res_data.columns[1:].tolist()
    res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
    res_data[res_data == -1] = 0
    gts_data[gts_data == -1] = 0

    metrics = compute_mlc(gts_data, res_data, label_set)
    pprint(metrics)

    metrics = compute_mlc(res_data, res_data, label_set)
    pprint(metrics)

    metrics = compute_mlc(gts_data, gts_data, label_set)
    pprint(metrics)

    metrics = compute_mlc_for_std(gts_data, res_data, label_set)
    pprint(metrics)



if __name__ == '__main__':
    main()





""" r2gen IU X-Ray (Questioning)
{'F1_MACRO': 0.0673843131753125,
 'F1_MICRO': 0.5580110497237568,
 'PRECISION_MACRO': 0.06820025811939916,
 'PRECISION_MICRO': 0.5297202797202797,
 'RECALL_MACRO': 0.08457033388067871,
 'RECALL_MICRO': 0.5894941634241245}
"""

""" r2gen MIMIC-CXR
{'F1_MACRO': 0.24132174375614085,
 'F1_MICRO': 0.39355182520650145,
 'PRECISION_MACRO': 0.3498752926192972,
 'PRECISION_MICRO': 0.46271929824561403,
 'RECALL_MACRO': 0.23484551429160308,
 'RECALL_MICRO': 0.3423736671302735}
 {'F1_MACRO': 0.20942392652537978,
 'F1_MICRO': 0.20942392652537978,
 'PRECISION_MACRO': 0.2413462517620474,
 'PRECISION_MICRO': 0.2413462517620474,
 'RECALL_MACRO': 0.21093616522691652,
 'RECALL_MICRO': 0.21093616522691652}
"""


""" cvt2 IU X-Ray (Questioning)
{'F1_MACRO': 0.04815409309791332,
 'F1_MICRO': 0.5434782608695652,
 'PRECISION_MACRO': 0.03631961259079903,
 'PRECISION_MICRO': 0.5084745762711864,
 'RECALL_MACRO': 0.07142857142857142,
 'RECALL_MICRO': 0.5836575875486382}
"""

""" cvt2 MIMIC-CXR
{'F1_MACRO': 0.2541386819142049,
 'F1_MICRO': 0.43610425971601896,
 'PRECISION_MACRO': 0.36785503694683175,
 'PRECISION_MICRO': 0.504783950617284,
 'RECALL_MACRO': 0.25152016703367375,
 'RECALL_MICRO': 0.38387513202675744}
 
 {'F1_MACRO': 0.22802381298746385,
 'F1_MICRO': 0.22802381298746385,
 'PRECISION_MACRO': 0.23100374709541965,
 'PRECISION_MICRO': 0.23100374709541965,
 'RECALL_MACRO': 0.256635775429223,
 'RECALL_MICRO': 0.256635775429223
"""