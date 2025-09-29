import pandas as pd
import glob
import numpy as np
import argparse


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--dataset', type=str, default="mimic",
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--test_path', type=str, default="sample_ckpt/convnext_mimic/",
                        help='Store the latest checkpoint in each epoch')

    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    ## Sorted cluster file by filename
    result_csv_list = []

    for dirname in glob.glob(args.test_path + "/1*"):
        print(dirname)
        result_csv_list.append(dirname + '/test_result.csv')

    print(result_csv_list)

    global_length = 0

    for i in range(len(result_csv_list)):
        data_csv = pd.read_csv(result_csv_list[i])
        print(data_csv.columns)
        print(len(data_csv))
        global_length = len(data_csv)
        data_csv.sort_values(['filename'], axis=0, ascending=[False], inplace=True)
        data_csv.to_csv(result_csv_list[i].replace(".csv", "_sorted.csv"), index=False)

    ## Start cluster combine
    result_csv_list = []

    for dirname in glob.glob(args.test_path + "/1*"):
        print(dirname)
        result_csv_list.append(dirname + '/test_result_sorted.csv')

    total_predict = []
    total_gt = []

    ## Pre-check for calibrate the file name
    counter = 0

    for i in range(1, len(result_csv_list)):
        first_csv = pd.read_csv(result_csv_list[i])
        second_csv = pd.read_csv(result_csv_list[i - 1])
        for s1 in range(global_length):
            if first_csv['filename'][s1] == second_csv['filename'][s1]:
                counter += 1

    if counter == global_length * (len(result_csv_list) - 1):
        print("File Calibrate")
    else:
        print("File not calibrate, please check")

    ## Add csv Column
    column_name = ['filename']

    case_final = pd.DataFrame(columns=column_name, data=list(zip(list(first_csv['filename']))))

    for i in range(len(result_csv_list)):
        column_name.append(result_csv_list[i].split('/')[-2] + "_predict_result")
        column_name.append(result_csv_list[i].split('/')[-2] + "_gt")

    column_name.append("All_predict_result")
    column_name.append("All_gt")
    print(column_name)

    ## Add Single file

    for i in range(len(result_csv_list)):
        first_csv = pd.read_csv(result_csv_list[i])

        for s1 in range(global_length):
            case_final[result_csv_list[i].split('/')[-2] + "_predict_result"] = first_csv["predict_result"][s1].replace(
                "[", "").replace("]", "").replace("\n", "").replace(" ", "").replace(
                ".", ",")[:-1]
            case_final[result_csv_list[i].split('/')[-2] + "_gt"] = first_csv["gt"][s1].replace("[", "").replace("]",
                                                                                                                 "").replace(
                "tensor(", "").replace(")", "").replace(
                "\n", "").replace(" ", "").replace("'", "")

    temp_csv = pd.read_csv(result_csv_list[0])
    for s1 in range(global_length):
        temp_csv["predict_result"][s1] = temp_csv["predict_result"][s1].replace("[", "").replace(
            "]", "").replace("\n", "").replace(" ", "").replace(
            ".", ",")[:-1]
        temp_csv["gt"][s1] = temp_csv["gt"][s1].replace("[", "").replace("]", "").replace(
            "tensor(", "").replace(")", "").replace(
            "\n", "").replace(" ", "").replace("'", "")

    for i in range(1, len(result_csv_list)):
        first_csv = pd.read_csv(result_csv_list[i])

        for s1 in range(global_length):
            temp_csv['predict_result'][s1] = temp_csv['predict_result'][s1] + "," + first_csv['predict_result'][s1]
            temp_csv['predict_result'][s1] = temp_csv['predict_result'][s1].replace("[", "").replace(
                "]", "").replace("\n", "").replace(" ", "").replace(
                ".", ",")[:-1]
            temp_csv['gt'][s1] = temp_csv['gt'][s1] + "," + first_csv['gt'][s1]
            temp_csv['gt'][s1] = temp_csv['gt'][s1].replace("[", "").replace("]", "").replace(
                "tensor(", "").replace(")", "").replace(
                "\n", "").replace(" ", "").replace("'", "")

    case_final["All_predict_result"] = list(temp_csv["predict_result"])
    case_final["All_gt"] = list(temp_csv["gt"])

    ## Only for checking file
    #print(case_final["100_predict_result"][11])
    #print(case_final["100_gt"][11])
    #print(len(case_final["100_predict_result"][11].split(',')))
    #print(len(case_final["100_gt"][11].split(',')))

    #print(case_final["10_predict_result"][11])
    #print(case_final["10_gt"][11])
    #print(len(case_final["10_predict_result"][11].split(',')))
    #print(len(case_final["10_gt"][11].split(',')))

    #print(case_final["1000_predict_result"][11])
    #print(case_final["1000_gt"][11])
    #print(len(case_final["1000_predict_result"][11].split(',')))
    #print(len(case_final["1000_gt"][11].split(',')))

    #print(case_final["All_predict_result"][11])
    #print(case_final["All_gt"][11])
    #print(len(case_final["All_predict_result"][11].split(',')))
    #print(len(case_final["All_gt"][11].split(',')))

    case_final.to_csv(args.test_path + "/all_cluster_result.csv",
        index=False)





