# General Imports
import os
import sys
import re
import torch
import pandas as pd
from tqdm import tqdm
import logging

# Libraries for Metrics
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score

# Sentence Transformers and Model
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Annotation
from statannotations.Annotator import Annotator

# Download necessary NLTK resources
import nltk
# nltk.download('punkt')

# Set up logging configuration (optional)
logging.basicConfig(level=logging.INFO)

# Load dataset
df = pd.read_csv('dataset/dataset_aug_train_all_new.csv')  

# Model Parameters
MODEL_NAME = "meta-llama/Llama-3.2-1B"
BATCH_SIZE = 32  # Adjust this based on your GPU memory capacity

# Load tokenizer and model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.to(device)

# Define prompt template
PROMPT_TEMPLATE = """
You are evaluating the quality of a sentence augmentation.
Given the original sentence and its augmented version, provide three scores:
1. **Semantic Preservation (0-100):** How well does the augmented sentence preserve the original meaning?
2. **Error Severity (0-100):** How incorrect is the augmentation? (Higher means more incorrect)
3. **Diversity Score (0-100):** How different is the augmentation structurally while still being valid?

Return scores in the format: `Semantic: X, Error: Y, Diversity: Z`.

Original: "{original}"
Augmented: "{augmented}"

Scores:
"""

# Function to generate scores for a batch using LLaMA
def get_scores_batch(originals, augmenteds):
    # Ensure tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # or tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    prompts = [PROMPT_TEMPLATE.format(original=o, augmented=a) for o, a in zip(originals, augmenteds)]

    # Set padding to left for decoder-only models
    tokenizer.padding_side = "left"
    
    # Tokenizing the prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # Set pad_token_id explicitly to eos_token_id to avoid the warning
    pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            pad_token_id=pad_token_id
        )
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract scores using regex for each response
    scores = []
    for response in responses:
        match = re.search(r"Semantic:\s*([\d.]+),\s*Error:\s*([\d.]+),\s*Diversity:\s*([\d.]+)", response)
        if match:
            scores.append((float(match.group(1)), float(match.group(2)), float(match.group(3))))
        else:
            scores.append((None, None, None))
    
    return scores

# Identify augmentation columns
augmentation_cols = [col for col in df.columns if col.startswith("augment_")]

# Initialize results list
results = []

# Process augmentations in batches
for col in augmentation_cols:
    original_texts = df["text"].tolist()
    augmented_texts = df[col].tolist()
    
    # Filter out invalid entries
    valid_pairs = [(o, a) for o, a in zip(original_texts, augmented_texts) 
                  if isinstance(a, str) and not pd.isna(a)]
    if not valid_pairs:
        continue
    
    original_batch = [pair[0] for pair in valid_pairs]
    augmented_batch = [pair[1] for pair in valid_pairs]
    
    # Process in batches
    for i in tqdm(range(0, len(valid_pairs), BATCH_SIZE), desc=f"Processing {col}"):
        batch_orig = original_batch[i:i + BATCH_SIZE]
        batch_aug = augmented_batch[i:i + BATCH_SIZE]
        
        batch_scores = get_scores_batch(batch_orig, batch_aug)
        
        # Append results with the column name
        for j, (semantic, error, diversity) in enumerate(batch_scores):
            results.append({
                "sentence_id": i + j,
                "augmentation_column": col,  # Column name added here
                "original_text": batch_orig[j],  # Appending original text
                "augmented_text": batch_aug[j],  # Appending augmented text
                "semantic_score": semantic,
                "error_score": error,
                "diversity_score": diversity
            })
# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('dataset/BERTEval.csv', index=False)

logging.info("Results saved to dataset/BERTEval.csv")
