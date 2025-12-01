import os
import pandas as pd

# -----------------------------------------------------------------------------
# Path Definitions
# -----------------------------------------------------------------------------
def get_mimic_files(data_dir):
    """
    Returns specific file paths for MIMIC-CXR based on your naming convention.
    """
    train = os.path.join(data_dir, "filter_general_predict_keywords_mimic_cxr_train.csv")
    val   = os.path.join(data_dir, "filter_general_predict_keywords_mimic_cxr_val.csv")
    test  = os.path.join(data_dir, "filter_general_predict_keywords_mimic_cxr_test.csv")
    return train, val, test

def get_iuxray_files(data_dir):
    """
    Returns specific file paths for IU-Xray.
    """
    train = os.path.join(data_dir, "filter_general_predict_keywords_iuxray_train.csv")
    val   = os.path.join(data_dir, "filter_general_predict_keywords_iuxray_val_v2.csv")
    test  = os.path.join(data_dir, "filter_general_predict_keywords_iuxray_test_v2.csv")
    return train, val, test

def get_custom_files(data_dir):
    """
    Returns paths for a custom dataset.
    Expects: train.csv, val.csv, test.csv
    """
    train = os.path.join(data_dir, "train.csv")
    val   = os.path.join(data_dir, "val.csv")
    test  = os.path.join(data_dir, "test.csv")
    return train, val, test

def get_dataset_paths(dataset_name, data_dir):
    """
    Switch function to get the correct paths based on dataset name.
    """
    if dataset_name == 'mimic':
        return get_mimic_files(data_dir)
    elif dataset_name == 'iuxray':
        return get_iuxray_files(data_dir)
    elif dataset_name == 'custom':
        return get_custom_files(data_dir)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------
def load_and_process_data(file_path, source_col, target_col):
    """
    Reads a CSV and formats it for SimpleT5 training.
    
    Args:
        file_path (str): Path to the CSV file.
        source_col (str): Column name for input keywords.
        target_col (str): Column name for ground truth report.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Case_num', 'source_text', 'target_text']
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Validate columns exist
    if source_col not in df.columns:
        raise ValueError(f"Source column '{source_col}' not found. Available: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    # Format inputs with the prefix expected by T5
    # "Keyword to Text: [keywords]"
    source_list = ["Keyword to Text: " + str(x) for x in df[source_col]]
    target_list = list(df[target_col])
    
    # Handle Case Numbers (Create dummy index if Case_num doesn't exist)
    if 'Case_num' in df.columns:
        case_list = list(df['Case_num'])
    else:
        case_list = [f"case_{i}" for i in range(len(df))]
        
    # Create the DataFrame expected by SimpleT5
    processed_df = pd.DataFrame({
        "Case_num": case_list,
        "source_text": source_list,
        "target_text": target_list
    })
    
    return processed_df