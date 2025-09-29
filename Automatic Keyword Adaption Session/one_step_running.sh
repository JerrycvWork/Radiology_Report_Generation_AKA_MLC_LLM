# Step 1: Construct RadLex Dictionary

python dictionary_extraction.py ## This dictionary extraction is actually what we done in manuscript.
#python RadLex_Reconstruction.py ## Please note that this will generate different dictionary compared with first command. In following all code, we use the dictionary from first command.

## Step 2: Extract keywords from Radiology report based on RadLex Dictionary

python keyword_extraction.py --dataset "iuxray" --split "train" --input_csv "Dataset_report/iu_xray/iuxray_train_full.csv" --output_csv "generate_keyword/iuxray_train_full.csv"
python keyword_extraction.py --dataset "iuxray" --split "test" --input_csv "Dataset_report/iu_xray/iuxray_test_full.csv" --output_csv "generate_keyword/iuxray_test_full.csv"
python keyword_extraction.py --dataset "iuxray" --split "val" --input_csv "Dataset_report/iu_xray/iuxray_val_full.csv" --output_csv "generate_keyword/iuxray_val_full.csv"
python keyword_extraction.py --dataset "mimic_cxr" --split "train" --input_csv "Dataset_report/mimic_cxr/mimic_train_full.csv" --output_csv "generate_keyword/mimic_train_full.csv"
python keyword_extraction.py --dataset "mimic_cxr" --split "test" --input_csv "Dataset_report/mimic_cxr/mimic_test_full.csv" --output_csv "generate_keyword/mimic_test_full.csv"
python keyword_extraction.py --dataset "mimic_cxr" --split "val" --input_csv "Dataset_report/mimic_cxr/mimic_val_full.csv" --output_csv "generate_keyword/mimic_val_full.csv"

## Step 3: Extract Frequency cluster based on 2 datasets

python frequency_analysis_iuxray.py --train_csv "generate_keyword/iuxray_train_full.csv" --test_csv "generate_keyword/iuxray_test_full.csv" --val_csv "generate_keyword/iuxray_val_full.csv" --output_dir "Encode_Dataset/"
python frequency_analysis_mimic_cxr.py --train_csv "generate_keyword/mimic_train_full.csv" --test_csv "generate_keyword/mimic_test_full.csv" --val_csv "generate_keyword/mimic_val_full.csv" --output_dir "Encode_Dataset/"


