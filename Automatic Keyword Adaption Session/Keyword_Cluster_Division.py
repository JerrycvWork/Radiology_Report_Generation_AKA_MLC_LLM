## Num of IU Xray is 3
## And Num of the MIMIC-CXR is 5
import pandas as pd


# Set by manual
iu_xray_cluster=3
mimic_cxr_cluster=5

current_dataset="iuxray"  #"iuxray" "mimic"

if current_dataset=="iuxray":
    frequency_csv = pd.read_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/"
        r"Automatic Keyword Adaption Session/keyword_frequency/iuxray_total_keyword_frequency_level1_train.csv")

    train_report_csv=pd.read_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/"
        r"Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_iuxray_train.csv")

    test_report_csv=pd.read_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/"
        r"Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_iuxray_test_v2.csv")

    val_report_csv=pd.read_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/"
        r"Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_iuxray_val_v2.csv")

    frequency_list=[]
    frequency_value = []

    ##(Sorted the list first)

    for s1 in range(1,iu_xray_cluster+1):
        frequency_list.append([])
        frequency_value.append([])

    for i in range(len(list(frequency_csv["frequency"]))):
       for s1 in range(1,iu_xray_cluster+1):
          if int(list(frequency_csv["frequency"])[i]) > 10**s1 and int(list(frequency_csv["frequency"])[i]) < 10**(s1+1):
             frequency_list[s1-1].append(list(frequency_csv["keyword"])[i].replace("'",""))
             frequency_value[s1-1].append(list(frequency_csv["frequency"])[i])

    #print(frequency_list)
    #print(frequency_value)

    #print(frequency_value[-1])
    #print(frequency_value[-2])
    #print(frequency_value[-3])

    for s2 in range(1,iu_xray_cluster+1):
        csv_filename=str(10**s2)
        # Frequency value=frequency_value[s2-1]
        # Frequency word=frequency_list[s2-1]
        case_final = pd.DataFrame(
            columns=['Keyword', 'Keyword_Frequency'], data=list(zip(list(frequency_list[s2-1]), list(frequency_value[s2-1]))))
        case_final.to_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/IUXRay_Frequency_List_"+ csv_filename+'.csv',index=False)





elif current_dataset=="mimic":

    frequency_csv = pd.read_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/"
        r"Automatic Keyword Adaption Session/keyword_frequency/mimic_cxr_total_keyword_frequency_level1_train.csv")

    train_report_csv=pd.read_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/"
        r"Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_mimic_cxr_train_revise.csv")

    test_report_csv=pd.read_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/"
        r"Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_mimic_cxr_test_revise.csv")

    val_report_csv=pd.read_csv(
        r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/"
        r"Automatic Keyword Adaption Session/Full_Data/filter_general_predict_keywords_mimic_cxr_val_revise.csv")

    train_encoding_str = [] * mimic_cxr_cluster

    frequency_list=[]
    frequency_value = []

    ##(Sorted the list first)

    for s1 in range(1,mimic_cxr_cluster+1):
        frequency_list.append([])
        frequency_value.append([])

    for i in range(len(list(frequency_csv["frequency"]))):
       for s1 in range(1,mimic_cxr_cluster+1):
          if int(list(frequency_csv["frequency"])[i]) > 10**s1 and int(list(frequency_csv["frequency"])[i]) < 10**(s1+1):
             frequency_list[s1-1].append(list(frequency_csv["keyword"])[i].replace("'",""))
             frequency_value[s1-1].append(list(frequency_csv["frequency"])[i])

    print(frequency_list)
    print(frequency_value)

    for s2 in range(1,mimic_cxr_cluster+1):
        csv_filename=str(10**s2)
        # Frequency value=frequency_value[s2-1]
        # Frequency word=frequency_list[s2-1]
        case_final = pd.DataFrame(
            columns=['Keyword', 'Keyword_Frequency'], data=list(zip(list(frequency_list[s2-1]), list(frequency_value[s2-1]))))
        case_final.to_csv(r"/home/htihe/Radiology_Report_Generation_Paper_Tag/Radiology_Report_Generation_AKA_MLC_LLM/Automatic Keyword Adaption Session/Encode_Frequency_List/mimic_cxr_Frequency_List_"+ csv_filename+'.csv',index=False)



