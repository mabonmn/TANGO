import argparse
import ast
import os
import random
import sys
import re
import time
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.marian import MarianMTModel
from concurrent.futures import ThreadPoolExecutor, as_completed

import nest_asyncio
nest_asyncio.apply()

import asyncio
from googletrans import Translator, LANGUAGES


class TextDataset(Dataset):
    def __init__(self, tokenizer, original_data_path=None, text_data_list=None):
        self.tokenizer = tokenizer
        if original_data_path:
            self.df = pd.read_csv(original_data_path)
            self.text_data_list = self.df['text'].tolist()
            self.text_num_list = [1] * len(self.text_data_list)
        else:
            self.text_data_list = text_data_list
            self.text_num_list = [1] * len(text_data_list)
    
    def __len__(self):
        return len(self.text_data_list)
    
    def __getitem__(self, idx):
        return self.text_data_list[idx]
    
    def collate_fn(self, batch):
        return self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)


class BackTranslation:
    def __init__(self, lang="de"):
        self.lang = lang
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        try:
            print("USING:", self.device)
            self.en_lang_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
            self.lang_en_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en")
            self.en_lang_translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}").to(self.device)
            self.lang_en_translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en").to(self.device)
        except Exception as e:
            print(f"Warning: Could not load models for language {lang}: {str(e)}")
            self.en_lang_tokenizer = None

    def do_back_translation(self, original_data_path, batch_size, temperature, **generate_kwargs):
        if not self.en_lang_tokenizer:  # Skip if model loading failed
            return None, None
            
        assert len(temperature) <= 2
        temp1, temp2 = (temperature[0], temperature[0]) if len(temperature) == 1 else temperature
        pandas_text_dataset = TextDataset(self.en_lang_tokenizer, original_data_path=original_data_path)
        dataloader = DataLoader(
            pandas_text_dataset, shuffle=False, drop_last=False, num_workers=4, 
            batch_size=batch_size, collate_fn=pandas_text_dataset.collate_fn
        )
        text_num_list = pandas_text_dataset.text_num_list
        lang_out_list = []
        for batch in tqdm(dataloader, desc=f"Translating to {self.lang}"):
            lang_out = self.en_lang_translator.generate(**batch.to(self.device), temperature=temp1, **generate_kwargs)
            for out in lang_out:
                lang_out_list.append(self.en_lang_tokenizer.decode(out).replace("<pad>", "").replace("</s>", "").strip())
        
        lang_text_dataset = TextDataset(self.lang_en_tokenizer, text_data_list=lang_out_list)
        dataloader = DataLoader(
            lang_text_dataset, shuffle=False, drop_last=False, num_workers=4, 
            batch_size=batch_size, collate_fn=lang_text_dataset.collate_fn
        )
        en_out_list = []
        for batch in tqdm(dataloader, desc=f"Translating back from {self.lang}"):
            en_out = self.lang_en_translator.generate(**batch.to(self.device), temperature=temp2, **generate_kwargs)
            for out in en_out:
                en_out_list.append(self.lang_en_tokenizer.decode(out).replace("<pad>", "").replace("</s>", "").strip())
        
        text_augment_list = []
        start = 0
        for text_num in text_num_list:
            text_augment_list.append(en_out_list[start : start + text_num])
            start += text_num
            
        return lang_out_list, text_augment_list

    def cleanup(self):
        """Clean up models and free CUDA memory."""
        if hasattr(self, 'en_lang_translator') and self.en_lang_translator is not None:
            del self.en_lang_translator
        if hasattr(self, 'lang_en_translator') and self.lang_en_translator is not None:
            del self.lang_en_translator
        if hasattr(self, 'en_lang_tokenizer') and self.en_lang_tokenizer is not None:
            del self.en_lang_tokenizer
        if hasattr(self, 'lang_en_tokenizer') and self.lang_en_tokenizer is not None:
            del self.lang_en_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def translate_text(text, translator, src='en', dest='en'):
    """Helper function to translate text with error handling and retry."""
    for _ in range(3):  # Retry up to 3 times
        try:
            if not isinstance(text, str) or not text.strip():
                return text
            translation = translator.translate(text, src=src, dest=dest)
            return translation.text if translation.text else text
        except Exception as e:
            print(f"Translation error: {e}. Retrying...")
            time.sleep(1)  # Brief delay before retry
    return text  # Fallback to original text if all retries fail


def googletrans_back_translate_with_suffix(df, lang_list, suffix="_gcp", text_col="text", num_threads=4, output_path=None):
    """Multithreaded Google Translate back-translation with incremental CSV updating."""
    translator = Translator()

    # Filter out languages that have already been processed
    if output_path and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_langs = {col.split('_')[1] for col in existing_df.columns if col.startswith(f'intermediate_') and col.endswith(suffix)}
        lang_list = [lang for lang in lang_list if lang not in processed_langs]
        print(f"Resuming Google Translate: Skipping {processed_langs}, remaining languages: {lang_list}")

    for lang in lang_list:
        intermediate_col = f"intermediate_{lang}{suffix}"
        augment_col = f"augment_{lang}{suffix}"
        
        # Prepare new columns as Series
        intermediate_series = pd.Series([None] * len(df), dtype="string")
        augment_series = pd.Series([None] * len(df), dtype="string")

        tasks = []
        indices = []
        original_texts = df[text_col].tolist()

        for idx, original_text in enumerate(original_texts):
            tasks.append((original_text, lang))
            indices.append(idx)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Forward translation
            future_to_idx = {
                executor.submit(translate_text, task[0], translator, src='en', dest=task[1]): idx 
                for idx, task in zip(indices, tasks)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(df), desc=f"GoogleTrans {lang} -> intermediate"):
                idx = future_to_idx[future]
                try:
                    intermediate_series[idx] = future.result()
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    intermediate_series[idx] = original_texts[idx]
            
            # Back translation
            future_to_idx = {
                executor.submit(translate_text, text, translator, src=lang, dest='en'): idx 
                for idx, text in zip(indices, intermediate_series)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(df), desc=f"GoogleTrans {lang} -> English"):
                idx = future_to_idx[future]
                try:
                    augment_series[idx] = future.result()
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    augment_series[idx] = original_texts[idx]
        
        # Add new columns using pd.concat to avoid fragmentation
        new_cols = pd.DataFrame({
            intermediate_col: intermediate_series,
            augment_col: augment_series
        })
        df = pd.concat([df, new_cols], axis=1)

        # Defragment the DataFrame
        df = df.copy()

        # Update the CSV with the new columns
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Updated CSV with results for language {lang} at {output_path}")

        # Flush CUDA memory after each language (though Google Translate doesn't use CUDA, added for consistency)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"Flushed CUDA memory after processing language {lang}")

    return df


async def run_multi_language_pipeline():
    languages_set1 = ['sq', 'ar', 'hy', 'eu', 'bg', 'ca', 'zh', 'cs', 'da', 'nl', 'et', 'fi', 'fr', 'gl', 'de', 'ht', 'hi', 'hu', 'is', 'id', 'ga', 'it', 'mk', 'ml', 'mt', 'mr', 'ru', 
                      'sk', 'es', 'sv', 'uk', 'ur', 'vi', 'cy'] 

    languages_set2 = ['bn', 'hr', 'ka', 'el', 'gu', 'he', 'ja', 'kn', 'kk', 'km', 'ko', 'lv', 'lt', 'ms', 'ne', 'no', 'fa', 'pl', 'pt', 'pa', 'ro', 'sr', 'sl', 'sw', 'ta', 'te', 'th', 'tr', 
                      'yi', 'zu']

    original_data_path = "dataset/train.csv"
    output_path = "dataset/dataset_aug_train_all_new.csv"
    print("FLAG")
    
    batch_size = qqq
    temperature = [1.0]
    num_beams = 5
    
    # Load original data
    original_df = pd.read_csv(original_data_path)
    print(f"Original dataset size: {len(original_df)}")

    # Check if output CSV exists and determine which languages have been processed
    processed_langs_set1 = set()
    processed_langs_set2 = set()
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        # Identify processed languages based on column names
        for col in existing_df.columns:
            if col.startswith('intermediate_') and col.endswith('_hels'):
                lang = col.split('_')[1]
                processed_langs_set1.add(lang)
            elif col.startswith('intermediate_') and col.endswith('_gcp'):
                lang = col.split('_')[1]
                processed_langs_set2.add(lang)
        print(f"Found existing CSV. Processed Helsinki-NLP languages: {processed_langs_set1}")
        print(f"Processed Google Translate languages: {processed_langs_set2}")
        original_df = existing_df  # Use the existing DataFrame as the starting point
    else:
        # Initialize CSV with original data if it doesn't exist
        original_df.to_csv(output_path, index=False)
        print(f"Initialized CSV with original data at {output_path}")

    # Filter languages to process only those that haven't been completed
    remaining_langs_set1 = [lang for lang in languages_set1 if lang not in processed_langs_set1]
    remaining_langs_set2 = [lang for lang in languages_set2 if lang not in processed_langs_set2]
    print(f"Remaining Helsinki-NLP languages to process: {remaining_langs_set1}")
    print(f"Remaining Google Translate languages to process: {remaining_langs_set2}")

    # Process Helsinki-NLP languages (languages_set1) with try-except for CUDA memory failures
    for lang in remaining_langs_set1:
        print(f"\nProcessing language: {lang} (Helsinki-NLP)")
        bt = BackTranslation(lang=lang)
        try:
            intermediate_texts, augmented_texts = bt.do_back_translation(
                original_data_path=original_data_path,
                batch_size=batch_size,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=True
            )
            if intermediate_texts and augmented_texts:
                # Flatten the lists since each sublist contains one item
                intermediate_texts = [text[0] if isinstance(text, list) else text for text in intermediate_texts]
                augmented_texts = [text[0] if isinstance(text, list) else text for text in augmented_texts]
                # Create new columns as Series
                intermediate_series = pd.Series(intermediate_texts, dtype="string")
                augment_series = pd.Series(augmented_texts, dtype="string")
                # Add new columns using pd.concat
                new_cols = pd.DataFrame({
                    f'intermediate_{lang}_hels': intermediate_series,
                    f'augment_{lang}_hels': augment_series
                })
                original_df = pd.concat([original_df, new_cols], axis=1)
            else:
                print(f"Skipping {lang} due to model loading failure")
                new_cols = pd.DataFrame({
                    f'intermediate_{lang}_hels': [None] * len(original_df),
                    f'augment_{lang}_hels': [None] * len(original_df)
                })
                original_df = pd.concat([original_df, new_cols], axis=1)
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA memory error while processing language {lang}: {str(e)}")
            print(f"Cleaning up and skipping to the next language...")
            bt.cleanup()  # Clean up CUDA memory
            # Add placeholder columns to indicate the language was skipped
            new_cols = pd.DataFrame({
                f'intermediate_{lang}_hels': [None] * len(original_df),
                f'augment_{lang}_hels': [None] * len(original_df)
            })
            original_df = pd.concat([original_df, new_cols], axis=1)
        except Exception as e:
            print(f"Unexpected error while processing language {lang}: {str(e)}")
            print(f"Cleaning up and skipping to the next language...")
            bt.cleanup()  # Clean up CUDA memory in case of other errors
            # Add placeholder columns to indicate the language was skipped
            new_cols = pd.DataFrame({
                f'intermediate_{lang}_hels': [None] * len(original_df),
                f'augment_{lang}_hels': [None] * len(original_df)
            })
            original_df = pd.concat([original_df, new_cols], axis=1)
        
        # Defragment the DataFrame
        original_df = original_df.copy()

        # Update the CSV with the new columns
        original_df.to_csv(output_path, index=False)
        print(f"Updated CSV with Helsinki-NLP results for {lang} at {output_path}")

        # Clean up and flush CUDA memory (already done in case of error, but ensure it's done for successful runs too)
        bt.cleanup()
        print(f"Flushed CUDA memory after processing language {lang}")

    # Process Google Translate languages (languages_set2)
    original_df = googletrans_back_translate_with_suffix(
        original_df, 
        lang_list=remaining_langs_set2, 
        suffix="_gcp", 
        text_col="text", 
        num_threads=10,
        output_path=output_path
    )
    
    print(f"\nFinal results saved to {output_path}")
    print("\nFinal Results Preview:")
    print(original_df.head())


async def main():
    torch.manual_seed(42)
    await run_multi_language_pipeline()


if __name__ == "__main__":
    asyncio.run(main())