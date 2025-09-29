## First of all, it should print out all the setting from Automatic Keyword Adaption.

python Multi_Label_Classification_Keyword_setting_print.py

## Train Function
python language_model_classification_train_test_asyloss.py --train_or_test "train" --net 'convnext' --Transformer 1 --output_dir "Multi_Label_Classification_Session/sample_ckpt/convnext_iuxray/" --dataset "iuxray"
python language_model_classification_train_test_asyloss.py --train_or_test "train" --net 'convnext' --Transformer 1 --output_dir "Multi_Label_Classification_Session/sample_ckpt/convnext_mimic/" --dataset "mimic"

## Test Function
python language_model_classification_train_test_asyloss.py --train_or_test "test" --net 'convnext' --test_ckpt_dir "Multi_Label_Classification_Session/sample_ckpt/convnext_iuxray/" --dataset "iuxray"
python language_model_classification_train_test_asyloss.py --train_or_test "test" --net 'convnext' --test_ckpt_dir "Multi_Label_Classification_Session/sample_ckpt/convnext_mimic/" --dataset "mimic"

## Combine Cluster of classification result
python classification_result_cluster_combine.py --dataset "iuxray" --test_path "Multi_Label_Classification_Session/sample_ckpt/convnext_iuxray/"
python classification_result_cluster_combine.py --dataset "mimic" --test_path "Multi_Label_Classification_Session/sample_ckpt/convnext_mimic/"

## Evaluation Function
python revised_eval_classification.py --dataset "iuxray" --test_path "Multi_Label_Classification_Session/sample_ckpt/convnext_iuxray/"
python revised_eval_classification.py --dataset "mimic" --test_path "Multi_Label_Classification_Session/sample_ckpt/convnext_mimic/"

## Generate Visible Keywords for radiology report generation
python classification_result_cluster_to_keyword.py --dataset "iuxray" --test_path "Multi_Label_Classification_Session/sample_ckpt/convnext_iuxray/"
python classification_result_cluster_to_keyword.py --dataset "mimic" --test_path "Multi_Label_Classification_Session/sample_ckpt/convnext_mimic/"
