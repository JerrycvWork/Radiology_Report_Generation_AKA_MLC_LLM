import pandas as pd
import glob

#for filename in glob.glob(r"/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/All_Radiology_Report/Ours/*/*.csv"):
filename="/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/All_Radiology_Report/Compare_Retrain/MIMIC-CXR/cvt2/test_reports_cvt2.csv"
print(filename)
total_csv=pd.read_csv(filename)
print(total_csv.columns)

## Generate Report=='Result'
## GT Report=='Ground-Truth'

## Copy Column==Report Impression

temp_result=list(total_csv["Report"])
temp_gt=list(total_csv["Ground-Truth"])

for i in range(len(temp_result)):
    temp_result[i]=temp_result[i].replace("Keyword to Text: ","").replace('"','').replace('[','').replace(']','')

for i in range(len(temp_gt)):
    temp_gt[i]=temp_gt[i].replace("Keyword to Text: ","").replace('"','').replace('[','').replace(']','')


case_final = pd.DataFrame(columns=["Report Impression"], data=temp_result)
case_final.to_csv(r"/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Radiology_Report_Transformation/"+filename.split("/")[-2]+"_"+filename.split("/")[-1].replace(".csv","_result.csv"), index=False)

case_final = pd.DataFrame(columns=["Report Impression"], data=temp_gt)
case_final.to_csv(r"/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/CheXbert-master/CheXbert-master/src/Radiology_Report_Transformation/"+filename.split("/")[-2]+"_"+filename.split("/")[-1].replace(".csv", "_groundtruth.csv"), index=False)




## R2Gen: Report,Ground-Truth