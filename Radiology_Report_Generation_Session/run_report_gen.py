import os
import time
import argparse
import torch
import pandas as pd
from simplet5 import SimpleT5

# Import dataset logic
from Dataset import get_dataset_paths, load_and_process_data
# Import new Evaluators
from evaluation import NLPEvaluator, ClinicalEvaluator

# -----------------------------------------------------------------------------
# 1. Configuration & Argument Parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Radiology Report Generation")
    
    # Mode & Data
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--dataset_name', type=str, choices=['mimic', 'iuxray', 'custom'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    
    parser.add_argument('--source_col', type=str, default='Level1_keywords')
    parser.add_argument('--target_col', type=str, default='Ground-Truth')
    
    # Model Config
    parser.add_argument('--model_type', type=str, default='t5')
    parser.add_argument('--model_name', type=str, default='t5-base')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--source_len', type=int, default=256)
    parser.add_argument('--target_len', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    # Test / Evaluation Config
    parser.add_argument('--checkpoint_path', type=str, default=None)
    
    # Clinical Metric Paths (New!)
    parser.add_argument('--chexbert_path', type=str, default=None, 
                        help="Path to chexbert.pth for clinical metrics")
    parser.add_argument('--radgraph_path', type=str, default=None, 
                        help="Path to model.tar.gz for RadGraph")
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# 2. Training Routine (Unchanged)
# -----------------------------------------------------------------------------
def train(args):
    train_path, val_path, _ = get_dataset_paths(args.dataset_name, args.data_dir)
    train_df = load_and_process_data(train_path, args.source_col, args.target_col)
    val_df = load_and_process_data(val_path, args.source_col, args.target_col)
    
    print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")
    
    model = SimpleT5()
    model.from_pretrained(model_type=args.model_type, model_name=args.model_name)
    
    output_subfolder = f"{args.dataset_name}_{args.model_type}_run"
    full_output_path = os.path.join(args.output_dir, output_subfolder)
    
    model.train(train_df=train_df,
                eval_df=val_df,
                source_max_token_len=args.source_len,
                target_max_token_len=args.target_len,
                batch_size=args.batch_size, 
                max_epochs=args.epochs, 
                use_gpu=True,
                outputdir=full_output_path)

# -----------------------------------------------------------------------------
# 3. Testing Routine (Updated with Metrics)
# -----------------------------------------------------------------------------
def test(args):
    if not args.checkpoint_path:
        raise ValueError("Error: --checkpoint_path is required for testing.")
        
    _, _, test_path = get_dataset_paths(args.dataset_name, args.data_dir)
    test_df = load_and_process_data(test_path, args.source_col, args.target_col)
    
    print(f"Loading Checkpoint: {args.checkpoint_path}")
    model = SimpleT5()
    model.load_model(model_type=args.model_type, model_dir=args.checkpoint_path, use_gpu=True)

    # --- Inference ---
    preds = []
    gts = []
    results = []
    
    print("Starting Inference...")
    start_time = time.time()
    
    for idx, row in test_df.iterrows():
        source = row['source_text']
        gt = row['target_text']
        
        with torch.no_grad():
            prediction = model.predict(source)[0]
        
        preds.append(prediction)
        gts.append(gt)
        results.append([row['Case_num'], source, prediction, gt])

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(test_df)}")

    print(f"Inference Time: {time.time() - start_time:.2f}s")

    # --- Save Basic CSV ---
    res_df = pd.DataFrame(results, columns=["Case_num", "Source", "Generated_Report", "Ground_Truth"])
    save_file = os.path.join(args.checkpoint_path, f"{args.dataset_name}_test_results.csv")
    res_df.to_csv(save_file, index=False)
    print(f"Basic results saved to: {save_file}")

    # --- 1. NLP Metrics (COCO) ---
    print("\n[Evaluation] Calculating NLP Metrics (BLEU, ROUGE, METEOR, CIDEr)...")
    #try:
    nlp_eval = NLPEvaluator()
    nlp_metrics = nlp_eval.evaluate(gts, preds)
        
    # Save metrics to a text file
    metric_file = os.path.join(args.checkpoint_path, "nlp_metrics.txt")
    with open(metric_file, "w") as f:
            for k, v in nlp_metrics.items():
                line = f"{k}: {v:.4f}"
                print(line)
                f.write(line + "\n")
    #except Exception as e:
        #print(f"NLP Evaluation Failed (Check Java/pycocoevalcap): {e}")

    # --- 2. Clinical Metrics (RadGraph/CheXbert) ---
    if args.chexbert_path and args.radgraph_path:
        print("\n[Evaluation] Calculating Clinical Metrics (RadGraph, CheXbert)...")
        try:
            clinical_eval = ClinicalEvaluator(args.chexbert_path, args.radgraph_path)
            
            clinical_out_csv = os.path.join(args.checkpoint_path, f"{args.dataset_name}_clinical_metrics.csv")
            
            # Pass the CSV path we just saved
            clinical_eval.run(save_file, save_file, clinical_out_csv)
            
            print("Clinical evaluation finished.")
        except Exception as e:
            print(f"Clinical Evaluation Failed: {e}")
    else:
        print("\n[Evaluation] Skipping Clinical Metrics (paths not provided).")

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)