from pprint import pprint

import pandas as pd

from metrics import compute_mlc, compute_mlc_for_std


def main():
    #res_path = "/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/IUX-Ray_result_result_labeled_reports.csv"
    #gts_path = "/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/IUX-Ray_result_groundtruth_labeled_reports.csv"
    #res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
    #res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

    #label_set = res_data.columns[1:].tolist()
    #res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
    #res_data[res_data == -1] = 0
    #gts_data[gts_data == -1] = 0

    #metrics = compute_mlc(gts_data, res_data, label_set)
    #pprint(metrics)

    #metrics = compute_mlc(res_data, res_data, label_set)
    #pprint(metrics)

    #metrics = compute_mlc(gts_data, gts_data, label_set)
    #pprint(metrics)

    res_path = "/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/Ours_MIMIC-CXR_result_testset_result_labeled_reports.csv"
    gts_path = "/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Extracted_report/Ours_MIMIC-CXR_result_testset_groundtruth_labeled_reports.csv"
    res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
    res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

    label_set = res_data.columns[1:].tolist()
    res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
    res_data[res_data == -1] = 0
    gts_data[gts_data == -1] = 0

    print(gts_data.shape)
    print(res_data.shape)

    metrics = compute_mlc(gts_data, res_data, label_set)
    pprint(metrics)

    metrics = compute_mlc(res_data, res_data, label_set)
    pprint(metrics)

    metrics = compute_mlc(gts_data, gts_data, label_set)
    pprint(metrics)

    metrics = compute_mlc_for_std(gts_data, res_data, label_set)
    pprint(metrics)

    #metrics = compute_mlc_for_std(res_data, res_data, label_set)
    #pprint(metrics)

    #metrics = compute_mlc_for_std(gts_data, gts_data, label_set)
    #pprint(metrics)




if __name__ == '__main__':
    main()





""" Ours IU X-Ray
{'F1_MACRO': 0.42004099253781063,
 'F1_MICRO': 0.7417475728155339,
 'PRECISION_MACRO': 0.5298567112711849,
 'PRECISION_MICRO': 0.764,
 'RECALL_MACRO': 0.3822142382727218,
 'RECALL_MICRO': 0.720754716981132}
"""

""" Ours MIMIC-CXR
{'F1_MACRO': 0.6190345744221577,
 'F1_MICRO': 0.7004476498383487,
 'PRECISION_MACRO': 0.6847060631705316,
 'PRECISION_MICRO': 0.7448102604786461,
 'RECALL_MACRO': 0.5782466313673335,
 'RECALL_MICRO': 0.661072644055862}
"""
