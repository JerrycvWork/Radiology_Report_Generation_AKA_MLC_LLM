import glob
import shutil


#/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/All_Radiology_Report/Comparison_of_Performance


import os
import glob

os.chdir('/home/htihe/PycharmProjects/CIBMProject/CE_Metrics_Calculation/All_Radiology_Report/')
result = glob.glob( '*/**.csv' )
print(result)
