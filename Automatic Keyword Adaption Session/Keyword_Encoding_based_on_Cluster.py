## Num of IU Xray is 3
## And Num of the MIMIC-CXR is 5
import pandas as pd


# Set by manual
iu_xray_cluster=3
mimic_cxr_cluster=5

current_dataset="mimic"  #"iuxray" "mimic"

if current_dataset=="iuxray":

    train_keyword_csv=pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_iuxray_train.csv")
    test_keyword_csv = pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_iuxray_test_v2.csv")
    val_keyword_csv= pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_iuxray_val_v2.csv")

    ## Cluster Setting loading

    frequency_csv_10=pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/IUXRay_Frequency_List_10.csv")
    frequency_csv_100 = pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/IUXRay_Frequency_List_100.csv")
    frequency_csv_1000 = pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/IUXRay_Frequency_List_1000.csv")

    list1000=list(frequency_csv_1000['Keyword'])
    list100 = list(frequency_csv_100['Keyword'])
    list10 = list(frequency_csv_10['Keyword'])

    print(list1000)
    print(list100)
    print(list10)

    ## Start to take the dataset

    train_string_list_1000 = []
    train_string_list_100 = []
    train_string_list_10 = []
    for i in range(len(list(train_keyword_csv['Level1_keywords']))):
        temp_str = list(train_keyword_csv['Level1_keywords'])[i].replace("'","")
        temp_list = temp_str.split(", ")
        connect_str = ""
        for s1 in range(len(list1000)):
            if list1000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        train_string_list_1000.append(connect_str)

        connect_str = ""
        for s1 in range(len(list100)):
            if list100[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        train_string_list_100.append(connect_str)

        connect_str = ""
        for s1 in range(len(list10)):
            if list10[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        train_string_list_10.append(connect_str)

    case_final = pd.DataFrame(
        columns=['Case_num', 'Level1_keywords', '1000_coding_str', "100_coding_str", "10_coding_str"], data=list(
            zip(list(train_keyword_csv['Case_num']), list(train_keyword_csv['Level1_keywords']), train_string_list_1000,
                train_string_list_100, train_string_list_10)))
    case_final.to_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Dataset/IUXray_train_frequency_data.csv",
        index=False)


    test_string_list_1000 = []
    test_string_list_100 = []
    test_string_list_10 = []
    for i in range(len(list(test_keyword_csv['Level1_keywords']))):
        temp_str = list(test_keyword_csv['Level1_keywords'])[i].replace("'","")
        temp_list = temp_str.split(", ")
        connect_str = ""
        for s1 in range(len(list1000)):
            if list1000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        test_string_list_1000.append(connect_str)

        connect_str = ""
        for s1 in range(len(list100)):
            if list100[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        test_string_list_100.append(connect_str)

        connect_str = ""
        for s1 in range(len(list10)):
            if list10[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        test_string_list_10.append(connect_str)

    case_final = pd.DataFrame(
        columns=['Case_num', 'Level1_keywords', '1000_coding_str', "100_coding_str", "10_coding_str"], data=list(
            zip(list(test_keyword_csv['Case_num']), list(test_keyword_csv['Level1_keywords']), test_string_list_1000,
                test_string_list_100, test_string_list_10)))
    case_final.to_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Dataset/IUXray_test_frequency_data.csv",
        index=False)


    val_string_list_1000 = []
    val_string_list_100 = []
    val_string_list_10 = []
    for i in range(len(list(val_keyword_csv['Level1_keywords']))):
        temp_str = list(val_keyword_csv['Level1_keywords'])[i].replace("'","")
        temp_list = temp_str.split(", ")
        connect_str = ""
        for s1 in range(len(list1000)):
            if list1000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        val_string_list_1000.append(connect_str)

        connect_str = ""
        for s1 in range(len(list100)):
            if list100[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        val_string_list_100.append(connect_str)

        connect_str = ""
        for s1 in range(len(list10)):
            if list10[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        val_string_list_10.append(connect_str)

    case_final = pd.DataFrame(
        columns=['Case_num', 'Level1_keywords', '1000_coding_str', "100_coding_str", "10_coding_str"], data=list(
            zip(list(val_keyword_csv['Case_num']), list(val_keyword_csv['Level1_keywords']), val_string_list_1000,
                val_string_list_100, val_string_list_10)))
    case_final.to_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Dataset/IUXray_val_frequency_data.csv",
        index=False)






elif current_dataset=="mimic":

    train_keyword_csv=pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_mimic_cxr_train_revise.csv")
    test_keyword_csv = pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_mimic_cxr_test_revise.csv")
    val_keyword_csv= pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_iuxray_val_v2.csv")

    ## Cluster Setting loading

    frequency_csv_10=pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/mimic_cxr_Frequency_List_10.csv")
    frequency_csv_100 = pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/mimic_cxr_Frequency_List_100.csv")
    frequency_csv_1000 = pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/mimic_cxr_Frequency_List_1000.csv")
    frequency_csv_10000 = pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/mimic_cxr_Frequency_List_10000.csv")
    frequency_csv_100000 = pd.read_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/mimic_cxr_Frequency_List_100000.csv")

    list100000 = list(frequency_csv_100000['Keyword'])
    list10000 = list(frequency_csv_10000['Keyword'])
    list1000=list(frequency_csv_1000['Keyword'])
    list100 = list(frequency_csv_100['Keyword'])
    list10 = list(frequency_csv_10['Keyword'])

    print(list100000)
    print(list10000)
    print(list1000)
    print(list100)
    print(list10)

    ## Start to take the dataset
    train_string_list_100000 = []
    train_string_list_10000 = []
    train_string_list_1000 = []
    train_string_list_100 = []
    train_string_list_10 = []
    for i in range(len(list(train_keyword_csv['Level1_keywords']))):
        temp_str = list(train_keyword_csv['Level1_keywords'])[i].replace("'","")
        temp_list = temp_str.split(", ")
        connect_str = ""

        for s1 in range(len(list100000)):
            if list100000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        train_string_list_100000.append(connect_str)

        for s1 in range(len(list10000)):
            if list10000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        train_string_list_10000.append(connect_str)

        for s1 in range(len(list1000)):
            if list1000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        train_string_list_1000.append(connect_str)

        connect_str = ""
        for s1 in range(len(list100)):
            if list100[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        train_string_list_100.append(connect_str)

        connect_str = ""
        for s1 in range(len(list10)):
            if list10[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        train_string_list_10.append(connect_str)

    case_final = pd.DataFrame(
        columns=['Case_num', 'Level1_keywords', '100000_coding_str', '10000_coding_str', '1000_coding_str', "100_coding_str", "10_coding_str"], data=list(
            zip(list(train_keyword_csv['Case_num']), list(train_keyword_csv['Level1_keywords']), train_string_list_100000, train_string_list_10000, train_string_list_1000,
                train_string_list_100, train_string_list_10)))
    case_final.to_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Dataset/mimic_train_frequency_data.csv",
        index=False)

    test_string_list_100000 = []
    test_string_list_10000 = []
    test_string_list_1000 = []
    test_string_list_100 = []
    test_string_list_10 = []
    for i in range(len(list(test_keyword_csv['Level1_keywords']))):
        temp_str = list(test_keyword_csv['Level1_keywords'])[i].replace("'","")
        temp_list = temp_str.split(", ")
        connect_str = ""

        for s1 in range(len(list100000)):
            if list100000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        test_string_list_100000.append(connect_str)

        for s1 in range(len(list10000)):
            if list10000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        test_string_list_10000.append(connect_str)

        for s1 in range(len(list1000)):
            if list1000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        test_string_list_1000.append(connect_str)

        connect_str = ""
        for s1 in range(len(list100)):
            if list100[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        test_string_list_100.append(connect_str)

        connect_str = ""
        for s1 in range(len(list10)):
            if list10[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        test_string_list_10.append(connect_str)

    case_final = pd.DataFrame(
        columns=['Case_num', 'Level1_keywords', '100000_coding_str', '10000_coding_str', '1000_coding_str', "100_coding_str", "10_coding_str"], data=list(
            zip(list(test_keyword_csv['Case_num']), list(test_keyword_csv['Level1_keywords']), test_string_list_100000, test_string_list_10000, test_string_list_1000,
                test_string_list_100, test_string_list_10)))
    case_final.to_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Dataset/mimic_test_frequency_data.csv",
        index=False)

    val_string_list_100000 = []
    val_string_list_10000 = []
    val_string_list_1000 = []
    val_string_list_100 = []
    val_string_list_10 = []
    for i in range(len(list(val_keyword_csv['Level1_keywords']))):
        temp_str = list(val_keyword_csv['Level1_keywords'])[i].replace("'","")
        temp_list = temp_str.split(", ")
        connect_str = ""

        for s1 in range(len(list100000)):
            if list100000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        val_string_list_100000.append(connect_str)

        for s1 in range(len(list10000)):
            if list10000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        val_string_list_10000.append(connect_str)

        for s1 in range(len(list1000)):
            if list1000[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        val_string_list_1000.append(connect_str)

        connect_str = ""
        for s1 in range(len(list100)):
            if list100[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        val_string_list_100.append(connect_str)

        connect_str = ""
        for s1 in range(len(list10)):
            if list10[s1] in temp_list:
                connect_str = connect_str + "1"
            else:
                connect_str = connect_str + "0"
        val_string_list_10.append(connect_str)

    case_final = pd.DataFrame(
        columns=['Case_num', 'Level1_keywords', '100000_coding_str', '10000_coding_str', '1000_coding_str', "100_coding_str", "10_coding_str"], data=list(
            zip(list(val_keyword_csv['Case_num']), list(val_keyword_csv['Level1_keywords']), val_string_list_100000, val_string_list_10000, val_string_list_1000,
                val_string_list_100, val_string_list_10)))
    case_final.to_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Dataset/mimic_val_frequency_data.csv",
        index=False)

