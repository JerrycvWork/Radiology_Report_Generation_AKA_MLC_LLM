## Setting print for put in the classification

import pandas as pd
import glob
import json

csv_data_folder="Encode_Dataset/"

## Iu X-Ray keyword cluster setting
iu_xray_frequency=[]
iu_xray_train_csv_path=""
iu_xray_test_csv_path=""
iu_xray_val_csv_path=""

for filename in glob.glob(csv_data_folder+"*iuxray_total*"):
    iu_xray_frequency.append(filename)

for filename in glob.glob(csv_data_folder+"*iuxray_train*"):
    iu_xray_train_csv_path=filename

for filename in glob.glob(csv_data_folder+"*iuxray_test*"):
    iu_xray_test_csv_path=filename

for filename in glob.glob(csv_data_folder+"*iuxray_val*"):
    iu_xray_val_csv_path=filename

#print(iu_xray_frequency)
print("IU XRay Cluster Number: ")
print(len(iu_xray_frequency))
print("IU XRay Train Csv Path: ")
print(iu_xray_train_csv_path)
print("IU XRay Test Csv Path: ")
print(iu_xray_test_csv_path)
print("IU XRay Valudation Csv Path: ")
print(iu_xray_val_csv_path)

iu_xray_total_frequency=[]
iu_xray_total_cluster=[]

for i in range(len(iu_xray_frequency)):
    print("Frequency Cluster")
    print(iu_xray_frequency[i].split("iuxray_total_")[1].split(".csv")[0])
    iu_xray_total_cluster.append(iu_xray_frequency[i].split("iuxray_total_")[1].split(".csv")[0])
    frequency_csv=pd.read_csv(iu_xray_frequency[i])
    print("Keyword Number: ")
    print(len(frequency_csv))
    iu_xray_total_frequency.append(len(frequency_csv))
    print("Highest Frequency: ")
    print(max(frequency_csv["frequency"]))


iu_xray_setting_json={
    "Train_csv_datapath": iu_xray_train_csv_path,
    "Test_csv_datapath": iu_xray_test_csv_path,
    "Val_csv_datapath": iu_xray_val_csv_path,
    "Keyword_Cluster_number": len(iu_xray_frequency),
    "Corresponding_Cluster_Frequency":iu_xray_total_cluster,
    "Corresponding_Cluster_keyword_Number": iu_xray_total_frequency
}

with open("Multi_Label_Classification_Session/iuxray_setting.json", "w") as f:
    json.dump(iu_xray_setting_json, f)

## MIMIC-CXR keyword cluster setting
mimic_frequency=[]
mimic_train_csv_path=""
mimic_test_csv_path=""
mimic_val_csv_path=""

for filename in glob.glob(csv_data_folder+"*mimic_total*"):
    mimic_frequency.append(filename)

for filename in glob.glob(csv_data_folder+"*mimic_train*"):
    mimic_train_csv_path=filename

for filename in glob.glob(csv_data_folder+"*mimic_test*"):
    mimic_test_csv_path=filename

for filename in glob.glob(csv_data_folder+"*mimic_val*"):
    mimic_val_csv_path=filename

#print(mimic_frequency)
print("MIMIC-CXR Cluster Number: ")
print(len(mimic_frequency))
print("MIMIC-CXR Train Csv Path: ")
print(mimic_train_csv_path)
print("MIMIC-CXR Test Csv Path: ")
print(mimic_test_csv_path)
print("MIMIC-CXR Valudation Csv Path: ")
print(mimic_val_csv_path)

mimic_total_frequency=[]
mimic_total_cluster=[]

for i in range(len(mimic_frequency)):
    print("Frequency Cluster")
    print(mimic_frequency[i].split("mimic_total_")[1].split(".csv")[0])
    mimic_total_cluster.append(mimic_frequency[i].split("mimic_total_")[1].split(".csv")[0])
    frequency_csv=pd.read_csv(mimic_frequency[i])
    print("Keyword Number: ")
    print(len(frequency_csv))
    mimic_total_frequency.append(len(frequency_csv))
    print("Highest Frequency: ")
    print(max(frequency_csv["frequency"]))


mimic_setting_json={
    "Train_csv_datapath": mimic_train_csv_path,
    "Test_csv_datapath": mimic_test_csv_path,
    "Val_csv_datapath": mimic_val_csv_path,
    "Keyword_Cluster_number": len(mimic_frequency),
    "Corresponding_Cluster_Frequency":mimic_total_cluster,
    "Corresponding_Cluster_keyword_Number": mimic_total_frequency
}

with open("Multi_Label_Classification_Session/mimic_setting.json", "w") as f:
    json.dump(mimic_setting_json, f)

