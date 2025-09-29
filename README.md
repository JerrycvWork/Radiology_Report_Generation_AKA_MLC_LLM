# Radiology_Report_Generation_AKA_MLC_LLM

The implemenetation of our proposed Radiology Report Generation with Automatic Keyword Adaption, Frequency-based Multi-Label Classification and Text-to-text Large Language Model.

[Paper link](https://www.sciencedirect.com/science/article/pii/S001048252500976X)

- [Environment Setting](#Environment-Setting)
- [Usage](#usage)
  - [Dataset and rearrange of the report](#Dataset-and-rearrange-of-the-report)
  - [Automatic Keyword Adaption Process](#Automatic-Keyword-Adaption-Process)
  - [Frequency-based Multi-Label Classification](#Frequency-based-Multi-Label-Classification)
  - [Using Keyword List to generate radiology report by Text-to-text Large Language Model](#Using-Keyword-List-to-generate-radiology-report-by-Text-to-text-Large-Language-Model)


## Environment Setting

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch in GPU version](https://pytorch.org/get-started/locally/)

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Dataset and rearrange of the report

In the experiments, we use the IU X-Ray dataset and MIMIC-CXR dataset.

1. For IU X-Ray dataset, please follow [Improving chest X-ray report generation by leveraging warm starting](https://github.com/aehrc/cvt2distilgpt2) to download the IU X-Ray images and their structured radiology reports. 

We also follow their split of Training set, Testing set and Validation set.

Also, please construct the radiology reports as the .csv file for the further usage.

The .csv file structure is shown in below:

| Image_id | Ground-Truth Report |
| ------------- | ------------- |
| CXR264_IM-1125 | "Heart XXXX, mediastinum, XXXX, bony structures and lung XXXX are unremarkable. No significant interval change compared to prior study, no XXXX infiltrates noted." |
| CXR3508_IM-1710 | "The heart is normal in size. The cardiomediastinal contours are stable. There are stable bilateral pleural effusions with partial right-sided loculation. Biapical scarring and pleural thickening appears stable. There is again right-sided superior hilar retraction and mild rightward XXXX deviation. No acute infiltrate is appreciated" | 
| CXR3224_IM-1524 | "The cardiomediastinal silhouette is normal in size and contour. No focal consolidation, pneumothorax or large pleural effusion. Negative for acute bone abnormality." |

2. For MIMIC-CXR dataset, please follow [Improving chest X-ray report generation by leveraging warm starting](https://github.com/aehrc/cvt2distilgpt2) first to download their structured radiology reports.

We also follow their split of Training set, Testing set and Validation set.

Also, please construct the radiology reports as the .csv file for the further usage.

The .csv file structure is shown in below:

| Image_id | Ground-Truth Report |
| ------------- | ------------- |
| f0707946-32499bba-77b6424d-f14642eb-587039a5 | "In comparison with the earlier study of this date, there are continued multifocal areas of consolidation with abscess formation especially at the right base.  Monitoring and support devices remain in place." |
| 67c9c5c6-f729ea08-a8ff4f27-2c8591bb-09775150 | "As compared to the previous radiograph, the size of the large right parahilar air-fluid level is slightly decreased.  Overall, the massive and predominantly central bilateral parenchymal opacities of mixed morphology are stable in extent and severity.  Unchanged normal size of the cardiac silhouette.  Unchanged absence of pleural effusions.  Unchanged mild elevation of the left hemidiaphragm." | 
| fa46f7c1-2f7b2152-3371f918-8971f374-e6405bae | "As compared to the previous radiograph, the size of the large right parahilar air-fluid level is slightly decreased.  Overall, the massive and predominantly central bilateral parenchymal opacities of mixed morphology are stable in extent and severity.  Unchanged normal size of the cardiac silhouette.  Unchanged absence of pleural effusions.  Unchanged mild elevation of the left hemidiaphragm." |

Then, please place all .csv files into the folder "Dataset_report/iu_xray/" and "Dataset_report/mimic_cxr/".

### Automatic Keyword Adaption Process

First of all, we provide the One-Key script for generating the data for classiification which is "Automatic Keyword Adaption Session/one_step_running.sh". Please check all the commands in this .sh file first. After running the script, the data for classification will be stored at "Automatic Keyword Adaption Session/Encode_Dataset".

Details of running:

1. Step 1: Construct RadLex Dictionary.

Please note that, we will use the first command for generating the dictionary in all experiments. The second command for generating the dictionary has not yet test. (It could run normally, but no any test based on this dictionary.)

2. Step 2: Extract keywords from Radiology report based on RadLex Dictionary

Each command will generate corresponding split in corresponding dataset. (Train/Test/Val) and (IU X-Ray/MIMIC-CXR).

3. Step 3: Extract Frequency cluster based on 2 datasets

Due to different report style, it could not write this step in 1 python file with all datasets. If you need to add another datasets, it should be rewritten in separate file for new dataset.



### Frequency-based Multi-Label Classification

Also, we provide the One-key script for generating the keywords for radiology report generation based on multi-label classification, which is "Multi_Label_Classification_Session\language_model_train_test_script_asyloss.sh". Please check all the commands in this .sh file. After running the script, the result for keywords using in radiology report generation will be stored at "all_cluster_result.csv" in correspongin network/dataset. The performance of classification will also stored at the same folders with filename "all_cluster_result_with_performance.csv".

Details of running:

1. Print out all the setting from Automatic Keyword Adaption.

Settings will be stored as .json file and it includes all parameters of data. (csv file, and cluster setting)/

2. Train Function/Test Function.

First of all, please check the "class hp" in file "Multi_Label_Classification_Session/language_model_classification_train_test_asyloss.py". It includes the parameters of training and test. Please put the image folder of each datasets (IU X-Ray and MIMIC-CXR) into the parameters correctly. 

Only the core functions (Training/Testing/Validation) writing. For other functions (Validation performance visualization/Keyword statistics visualization/Talos Optimization/etc), I still need the time for writing.


### Using Keyword List to generate radiology report by Text-to-text Large Language Model

Wait for the filling.

