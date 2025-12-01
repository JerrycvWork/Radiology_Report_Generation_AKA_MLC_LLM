import os
import json
import torch
import numpy as np
import pandas as pd
import re
import pickle
#from bert_score import BERTScorer
#from fast_bleu import BLEU as FastBLEU

# COCO Evaluation Imports
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Clinical Metric Imports (Assuming CXRMetric folder exists in root)
try:
    from CXRMetric.radgraph_evaluate_model import run_radgraph
except ImportError:
    print("Warning: CXRMetric module not found. Clinical metrics will fail if requested.")

# -----------------------------------------------------------------------------
# Part 1: Standard NLP Evaluator (COCO Style)
# -----------------------------------------------------------------------------
class NLPEvaluator:
    def __init__(self):
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

    def evaluate(self, ground_truth_list, candidate_list):
        """
        Args:
            ground_truth_list: List of ground truth strings.
            candidate_list: List of generated strings.
        Returns:
            dict: Evaluation results
        """
        # Format for pycocoevalcap: {index: [{'caption': text}]}
        gts = {i: [{'caption': str(t)}] for i, t in enumerate(ground_truth_list)}
        res = {i: [{'caption': str(t)}] for i, t in enumerate(candidate_list)}

        print("Tokenizing for NLP metrics (this requires Java)...")
        gts = self.tokenizer.tokenize(gts)
        res = self.tokenizer.tokenize(res)

        results = {}
        for scorer, method in self.scorer_list:
            print(f"Computing {method}...")
            score, scores = scorer.compute_score(gts, res)
            
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    results[m] = sc
            else:
                results[method] = score
        
        return results

# -----------------------------------------------------------------------------
# Part 2: Clinical Metrics Evaluator (RadGraph, CheXbert, etc.)
# -----------------------------------------------------------------------------
class ClinicalEvaluator:
    def __init__(self, chexbert_path, radgraph_path, cache_dir="./cache"):
        self.chexbert_path = chexbert_path
        self.radgraph_path = radgraph_path
        self.cache_dir = cache_dir
        
        # Hardcoded paths from your snippet, ideally passed as args or found relative
        self.normalizer_path = "CXRMetric/normalizer.pkl"
        self.composite_v0_path = "CXRMetric/composite_metric_model.pkl"
        self.composite_v1_path = "CXRMetric/radcliq-v1.pkl"
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def preprocess_reports(self, reports):
        """Preprocesses reports for Clinical Metrics"""
        return [list(filter(
            lambda val: val != "", str(elem).lower().replace(".", " .").split(" "))) 
            for elem in reports]

    def add_fast_bleu(self, gt_df, pred_df):
        """Computes BLEU-2 specifically for the RadCliQ calculation."""
        weights = {"bigram": (1/2., 1/2.)}
        pred_df["bleu_score"] = 0.0
        
        for i, row in gt_df.iterrows():
            gt_report = self.preprocess_reports([row["report"]])[0]
            # Match by Study ID
            pred_row = pred_df[pred_df["study_id"] == row["study_id"]]
            if len(pred_row) == 0: continue
            
            predicted_report = self.preprocess_reports([pred_row["report"].values[0]])[0]
            
            bleu = FastBLEU([gt_report], weights)
            score = bleu.get_score([predicted_report])["bigram"]
            
            idx = pred_row.index[0]
            pred_df.at[idx, "bleu_score"] = score[0]
        return pred_df

    def add_bertscore(self, gt_df, pred_df):
        print("Computing BERTScore...")
        refs = [re.sub(r' +', ' ', r) for r in gt_df["report"].tolist()]
        cands = [re.sub(r' +', ' ', r) for r in pred_df["report"].tolist()]

        scorer = BERTScorer(
            model_type="distilroberta-base",
            batch_size=64, # Reduced batch size for safety
            lang="en",
            rescale_with_baseline=True,
            idf=False, 
            idf_sents=refs
        )
        _, _, f1 = scorer.score(cands, refs)
        pred_df["bertscore"] = f1.numpy()
        return pred_df

    def add_semb_score(self, pred_df, gt_csv_path, pred_csv_path):
        print("Running CheXbert for s_emb...")
        pred_embed_path = os.path.join(self.cache_dir, "pred_embeddings.pt")
        gt_embed_path = os.path.join(self.cache_dir, "gt_embeddings.pt")

        # Call external script via OS command (as per original code)
        # Note: We assume CXRMetric/CheXbert/src/encode.py exists
        cmd_pred = f"python CXRMetric/CheXbert/src/encode.py -c {self.chexbert_path} -d {pred_csv_path} -o {pred_embed_path}"
        cmd_gt = f"python CXRMetric/CheXbert/src/encode.py -c {self.chexbert_path} -d {gt_csv_path} -o {gt_embed_path}"
        
        os.system(cmd_pred)
        os.system(cmd_gt)

        label_embeds = torch.load(gt_embed_path)
        pred_embeds = torch.load(pred_embed_path)
        
        # Calculate Cosine Similarity
        scores = []
        # Assuming sequential alignment if study_ids match
        # (This logic is simplified; usually you match by study_id key in the dict)
        for idx in range(len(pred_df)):
             # In your original code, it looped by sorted keys. 
             # Here we assume the CSVs are sorted/aligned.
             # If keys are study_ids:
             study_id = pred_df.iloc[idx]["study_id"]
             if study_id in label_embeds and study_id in pred_embeds:
                 v1 = label_embeds[study_id]
                 v2 = pred_embeds[study_id]
                 sim = (v1 * v2).sum() / (torch.norm(v1) * torch.norm(v2))
                 scores.append(sim.item())
             else:
                 scores.append(0.0)
                 
        pred_df["semb_score"] = scores
        return pred_df

    def run(self, gt_csv, pred_csv, out_csv):
        print("Starting Clinical Metric Evaluation...")
        
        # 1. Load Data
        gt = pd.read_csv(gt_csv)
        pred = pd.read_csv(pred_csv)

        # 2. Standardize Columns for CXRMetric tools
        # The tools expect 'study_id' and 'report'
        gt = gt.rename(columns={"Case_num": "study_id", "Ground_Truth": "report"})
        pred = pred.rename(columns={"Case_num": "study_id", "Generated_Report": "report"})
        
        # Ensure string
        gt['report'] = gt['report'].astype(str)
        pred['report'] = pred['report'].astype(str)
        
        # Save temp CSVs for the external scripts
        temp_gt_path = os.path.join(self.cache_dir, "temp_gt.csv")
        temp_pred_path = os.path.join(self.cache_dir, "temp_pred.csv")
        gt.to_csv(temp_gt_path, index=False)
        pred.to_csv(temp_pred_path, index=False)

        # 3. Compute Metrics
        # A. BLEU (FastBLEU for RadCliQ)
        eval_df = self.add_fast_bleu(gt, pred.copy())
        
        # B. BERTScore
        eval_df = self.add_bertscore(gt, eval_df)
        
        # C. s_emb (CheXbert)
        eval_df = self.add_semb_score(eval_df, temp_gt_path, temp_pred_path)
        
        # D. RadGraph
        print("Running RadGraph...")
        ent_path = os.path.join(self.cache_dir, "entities.json")
        rel_path = os.path.join(self.cache_dir, "relations.json")
        
        # Calling your provided imported function
        run_radgraph(temp_gt_path, temp_pred_path, self.cache_dir, self.radgraph_path, ent_path, rel_path)
        
        # Parse RadGraph Results
        with open(ent_path) as f: ent_scores = json.load(f)
        with open(rel_path) as f: rel_scores = json.load(f)
        
        rad_scores = []
        for sid in eval_df["study_id"]:
            sid = str(sid)
            s = 0
            if sid in ent_scores: s = float(ent_scores[sid][0]) # F1 score is index 0
            if sid in rel_scores: s = (s + float(rel_scores[sid][0])) / 2.0
            rad_scores.append(s)
        eval_df["radgraph_combined"] = rad_scores

        # 4. Composite Metric (RadCliQ)
        # Load models
        try:
            with open(self.normalizer_path, "rb") as f: normalizer = pickle.load(f)
            with open(self.composite_v1_path, "rb") as f: model_v1 = pickle.load(f)
            
            # Columns required: ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]
            cols = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]
            input_data = np.array(eval_df[cols])
            
            # Normalize? (Your logic had normalization for v0 but v1 usage varies. 
            # Assuming v1 takes raw or we use v0 logic. Using your v1 logic:)
            # Actually, usually RadCliQ v1 expects normalized input too? 
            # Following your snippet:
            # v1 code in your snippet: radcliq_v1_scores = composite_metric_v1_model.predict(input_data)
            # It seems v1 in your snippet didn't use the normalizer. 
            
            eval_df["RadCliQ-v1"] = model_v1.predict(input_data)
        except Exception as e:
            print(f"Skipping Composite Metric (RadCliQ) due to missing files: {e}")

        # Save
        eval_df.to_csv(out_csv, index=False)
        print(f"Clinical metrics saved to: {out_csv}")
        return eval_df