import argparse
import pandas as pd
import glob
from sklearn import metrics
import numpy as np
import json

def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--dataset', type=str, default="mimic",
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--test_path', type=str, default="/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Multi_Label_Classification_Session/sample_ckpt/convnext_mimic/",
                        help='Store the latest checkpoint in each epoch')

    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    with open(
            "/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Multi_Label_Classification_Session/" + args.dataset + "_setting.json",
            "r") as f:
        data = json.load(f)

    full_cluster_level = data["Corresponding_Cluster_Frequency"]
    full_cluster_level.append("All")

    csv_path = args.test_path+"/all_cluster_result.csv"

    summarization_list = []
    counter = 0

    summarization_list.append([])

    result_csv = pd.read_csv(csv_path)
    # print(result_csv.columns)

    Total_TP = 0
    Total_TN = 0
    Total_FP = 0
    Total_FN = 0

    Total_Acc = []
    Total_Sen = []  # TPR
    Total_Spe = []  # TNR
    Total_F1 = []
    Total_MCC = []

    Total_PPV = []
    Total_NPV = []

    f_name = []

    for i in range(len(full_cluster_level)):
        result_csv[full_cluster_level[i] + '_accuracy'] = [''] * len(result_csv)
        result_csv[full_cluster_level[i] + '_TPR'] = [''] * len(result_csv)
        result_csv[full_cluster_level[i] + '_TNR'] = [''] * len(result_csv)
        result_csv[full_cluster_level[i] + '_f1_score'] = [''] * len(result_csv)
        result_csv[full_cluster_level[i] + '_mcc'] = [''] * len(result_csv)

        result_csv[full_cluster_level[i] + '_PPV'] = [''] * len(result_csv)
        result_csv[full_cluster_level[i] + '_NPV'] = [''] * len(result_csv)

    for s1 in range(len(result_csv)):
        for i in range(len(full_cluster_level)):
            result_str = result_csv[full_cluster_level[i] + '_predict_result'][s1]
            gt_str = result_csv[full_cluster_level[i] + '_gt'][s1]

            result_numpy = np.fromstring(result_str, sep=",")
            # print(s1)
            gt_numpy = np.fromstring(gt_str, sep=",")

            if np.sum(gt_numpy) >= 0:
                c_m = metrics.confusion_matrix(gt_numpy, result_numpy)
                f1_score = metrics.f1_score(gt_numpy, result_numpy)
                mcc = metrics.matthews_corrcoef(gt_numpy, result_numpy)
                accuracy = metrics.accuracy_score(gt_numpy, result_numpy)

                if c_m.shape[0] > 1:
                    TP = c_m[1][1]
                    FN = c_m[1][0]
                    FP = c_m[0][1]
                    TN = c_m[0][0]
                elif c_m.shape[0] == 1:
                    TP = c_m[0][0]
                    FN = 0
                    FP = 0
                    TN = 0

                TPR = (TP) / (TP + FN)

                if TN > 0:
                    TNR = (TN) / (TN + FP)
                elif TN == 0 and FP > 0:
                    TNR = 0
                elif TN == 0 and FP == 0:
                    TNR = 1

                if TP == 0 and FP == 0:
                    PPV = 0
                else:
                    PPV = (TP) / (TP + FP)

                if TN > 0:
                    NPV = (TN) / (TN + FN)
                elif TN == 0 and FN > 0:
                    NPV = 0
                elif TN == 0 and FN == 0:
                    NPV = 0

                result_csv[full_cluster_level[i] + '_accuracy'][s1] = accuracy
                result_csv[full_cluster_level[i] + '_TPR'][s1] = TPR
                result_csv[full_cluster_level[i] + '_TNR'][s1] = TNR
                result_csv[full_cluster_level[i] + '_f1_score'][s1] = f1_score
                result_csv[full_cluster_level[i] + '_mcc'][s1] = mcc

                result_csv[full_cluster_level[i] + '_PPV'][s1] = PPV
                result_csv[full_cluster_level[i] + '_NPV'][s1] = NPV



    result_csv.to_csv(
        args.test_path+"/all_cluster_result_with_performance.csv",
        index=False)

    print("Performance Summary")

    for i in range(len(full_cluster_level)):
        print("Cluster ", full_cluster_level[i])

        print("mean")
        print("Accuracy  ", np.mean(result_csv[full_cluster_level[i] + '_accuracy']))
        print("Sensitivity  ", np.mean(result_csv[full_cluster_level[i] + '_TPR']))
        print("Specificity  ", np.mean(result_csv[full_cluster_level[i] + '_TNR']))
        print("F1-Score  ", np.mean(result_csv[full_cluster_level[i] + '_f1_score']))
        print("MCC  ", np.mean(result_csv[full_cluster_level[i] + '_mcc']))

        print("std")
        print("Accuracy  ", np.std(result_csv[full_cluster_level[i] + '_accuracy']))
        print("Sensitivity  ", np.std(result_csv[full_cluster_level[i] + '_TPR']))
        print("Specificity  ", np.std(result_csv[full_cluster_level[i] + '_TNR']))
        print("F1-Score  ", np.std(result_csv[full_cluster_level[i] + '_f1_score']))
        print("MCC  ", np.std(result_csv[full_cluster_level[i] + '_mcc']))



