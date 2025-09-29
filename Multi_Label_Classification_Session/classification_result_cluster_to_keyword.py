import argparse

import pandas as pd
import glob
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

    total_result_csv = pd.read_csv(
        args.test_path+"all_cluster_result.csv")

    print(total_result_csv.columns)

    with open(
            "/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Multi_Label_Classification_Session/" + args.dataset + "_setting.json",
            "r") as f:
        data = json.load(f)

    full_cluster_level = data["Corresponding_Cluster_Frequency"]

    for i in range(len(full_cluster_level)):
        total_result_csv[full_cluster_level[i] + "_encode_keyword_result"] = [''] * len(total_result_csv)
        total_result_csv[full_cluster_level[i] + "_encode_keyword_gt"] = [''] * len(total_result_csv)

    for i in range(len(full_cluster_level)):
        keyword_csv = pd.read_csv(
            "/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Alex_Code/Automatic Keyword Adaption Session_alex/Automatic Keyword Adaption Session_alex/Encode_Dataset/total_keyword_frequency_"+args.dataset+"_total_" +
            full_cluster_level[i] + ".csv")
        for s1 in range(len(total_result_csv)):
            #print(full_cluster_level[i])
            #print(total_result_csv[full_cluster_level[i]+"_predict_result"][s1])
            #print(total_result_csv[full_cluster_level[i] + "_gt"][s1])

            result_str = str(total_result_csv[full_cluster_level[i] + "_predict_result"][s1])
            gt_str = str(total_result_csv[full_cluster_level[i] + "_gt"][s1])

            for s2 in range(len(result_str.split(','))):
                if int(result_str.split(',')[s2]) == 1:
                    total_result_csv[full_cluster_level[i] + "_encode_keyword_result"][s1] = str(keyword_csv["keyword"][s2]) + ',' + total_result_csv[full_cluster_level[i] + "_encode_keyword_result"][s1]
                if int(gt_str.split(',')[s2]) == 1:
                    total_result_csv[full_cluster_level[i] + "_encode_keyword_gt"][s1] = str(keyword_csv["keyword"][s2])+ ',' +total_result_csv[full_cluster_level[i] + "_encode_keyword_gt"][s1]

    total_result_csv.to_csv(
        args.test_path+"all_cluster_result_encode_keyword.csv",
        index=False)

