from typing import Dict
import re
import csv
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from functools import partial

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

def process_medical_cot_example(
    example: Dict,
    tokenizer,
):
    # Map CSV columns to expected format
    question = example["progression_prompt"]  # Medical context/question
    thinking_trajectory = [example["llm_reasoning"]]  # Reasoning as single item list
    answer = example["target_medication"]  # Target medication

    # Replace target medication mentions in reasoning with mask
    # if answer:
    #     # Replace exact medication mentions with [MEDICATION] mask
    #     thinking_trajectory = [
    #         re.sub(re.escape(answer), '[MEDICATION]', t, flags=re.IGNORECASE) 
    #         for t in thinking_trajectory
    #     ]
    
    # Format the prompt
    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    
    # Apply chat template
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {
            "role": "assistant", 
            "content": "<start_of_turn>think\n" + "\n".join(thinking_trajectory).strip() + "\n<start_of_turn>answer\n" + answer.strip()
        }
    ], tokenize=False)
    return dict(text=text)

def medical_cot_sft(csv_file_path: str, output_path: str, num_proc: int):
    """Process medical CSV file with reasoning chains for SFT training"""
    
    # Load CSV file
    print(f"Loading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    # Filter out rows with missing essential data
    df = df.dropna(subset=['progression_prompt', 'llm_reasoning', 'target_medication'])
    print(f"Loaded {len(df)} valid examples after filtering")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    
    # Process examples
    process_example_map = partial(process_medical_cot_example, tokenizer=tokenizer)
    dataset = dataset.map(
        process_example_map,
        num_proc=num_proc,
        desc="Tokenizing medical SFT data",
        remove_columns=dataset.column_names,  # Remove all original columns, keep only 'text'
    )
    
    # Save the processed dataset as JSONL with proper character handling
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved tokenized dataset to: {output_path}")

if __name__ == "__main__":
    medical_cot_sft(
        csv_file_path="results/datasets/bmt_initial_progression_notes_with_reasoning_gemini.csv",
        output_path="results/datasets/bmt_initial_progression_notes_1b_tokenized.jsonl", 
        num_proc=20
    )