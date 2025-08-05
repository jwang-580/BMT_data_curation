import json
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field, validator
import os
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import sys

load_dotenv()

# Validate OpenAI API key
if not os.getenv('OAI_API_KEY'):
    print("Error: OpenAI API key not found. Please set the OAI_API_KEY environment variable.")
    sys.exit(1)

AVAILABLE_MODELS = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-o3-mini": "o3-mini-2025-01-31",
    "gpt-o1": "o1-2024-12-17"
}

class OpenAIConfig:
    def __init__(self, model_name: str):
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not in available models: {list(AVAILABLE_MODELS.keys())}")
        self.model_name = AVAILABLE_MODELS[model_name]
        self.client = OpenAI(api_key=os.getenv('OAI_API_KEY'))
        
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

def ensure_directory(directory: str):
    """Ensure the directory exists, create if it doesn't."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    except Exception as e:
        print(f"Error creating directory {directory}: {str(e)}")
        sys.exit(1)

def load_progress_notes():
    """Load progress notes with error handling."""
    input_file = 'results/datasets/combined_hospital_progress_medications_analysis.csv'
    try:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)
        return pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading progress notes: {str(e)}")
        sys.exit(1)

def load_assessment_plan(hadm_id):
    """Load assessment plan with error handling."""
    try:
        file_path = f'results/notes/a_p_{hadm_id}.json'
        if not os.path.exists(file_path):
            print(f"Warning: Assessment plan file not found: {file_path}")
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Error loading assessment plan for HADM_ID {hadm_id}: {str(e)}")
        return None

def process_notes_with_reasoning(openai_config: OpenAIConfig, output_dir: str, num_rows=None):
    """
    Process progress notes with OpenAI reasoning.
    
    Args:
        openai_config (OpenAIConfig): Configuration for OpenAI model
        output_dir (str): Directory to save output files
        num_rows (int, optional): Number of rows to process. If None, process all rows.
    """
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Load progress notes
    progress_notes = load_progress_notes()
    
    # Create reasoning dataset
    reasoning_dataset = progress_notes.copy()
    reasoning_dataset['llm_reasoning'] = None
    reasoning_dataset['processing_status'] = None  # Track processing status
    
    # Determine how many rows to process
    if num_rows is not None:
        rows_to_process = reasoning_dataset.head(num_rows)
        print(f"Processing {num_rows} rows out of {len(reasoning_dataset)} total rows")
    else:
        rows_to_process = reasoning_dataset
        print(f"Processing all {len(reasoning_dataset)} rows")

    print(f"Using OpenAI model: {openai_config.model_name}")

    successful_count = 0
    failed_count = 0

    for idx, row in rows_to_process.iterrows():
        try:
            assessment_plan = load_assessment_plan(row["HADM_ID"])
            if assessment_plan is None:
                reasoning_dataset.at[idx, 'processing_status'] = 'missing_assessment_plan'
                failed_count += 1
                continue
                
            cot_prompt = f"""
                You are provided with the following:
                1. A patient progress note
                2. The Assessment and Plan section of the note
                3. A specific medication that was started on the same day as the note
                Your task is to reason step-by-step like the treating physician that lead to the final conclusion that this medication needs to be started.

                Guidelines for Reasoning:

                1. Use a clinical reasoning tone (e.g. Alright, let's break this down...Okay, let's reconsider...Oh, wait a second, maybe we're missing something simpler...But hang on, what if...). As you are reasoning to the conclusion (the medication), the medication itself should not occur in the reasoning process.

                2.  Summarize the clinical course and intergrate clinical data (chief complaint, past medical history, recent events, vital signs, lab results, imaging findings, current medications, etc. MENTION SPECIFIC CLINICAL DATA!) in your reasoning; remember, imaging findings are usually important. 

                3. Review the Assessment and Plan section.
                    - It may explicitly state the reasoning for the medication.
                    - However, do not assume the reasoning is correct, assess it critically.
                    - DO NOT CITE ANYTHING FROM THE ASSESSMENT AND PLAN SECTION, but use it as if you come up with the reasoning yourself.
                
                4. Use this as the final sentence: Therefore, due to (summary of reasons), XXX (medication) was started today.

                ### Input Format:
                Progress Note:
                {row['Progress_note']}

                Assessment and Plan:
                {assessment_plan}

                Medication Started:
                {row['medication']}

                ### Output Format:
                Reasoning (Step-by-step clinical reasoning process):
                {{Your detailed reasoning trace}}
            """
                
            reasoning = openai_config.generate_response(cot_prompt)
            if reasoning is None:
                reasoning_dataset.at[idx, 'processing_status'] = 'api_error'
                failed_count += 1
                continue
                
            reasoning_dataset.at[idx, 'llm_reasoning'] = reasoning
            reasoning_dataset.at[idx, 'processing_status'] = 'success'
            successful_count += 1
            
            # Print progress every 5 records
            if (idx - rows_to_process.index[0]) % 5 == 0:
                print(f"Processed {idx - rows_to_process.index[0] + 1} out of {len(rows_to_process)} records...")
                
        except Exception as e:
            print(f"Error processing HADM_ID {row['HADM_ID']}: {str(e)}")
            reasoning_dataset.at[idx, 'processing_status'] = 'error'
            failed_count += 1

    # Save the reasoning dataset
    output_filename = os.path.join(output_dir, f'progress_notes_with_reasoning{"_" + str(num_rows) if num_rows else ""}.csv')
    try:
        reasoning_dataset.to_csv(output_filename, index=False)
        print("\nReasoning dataset saved successfully!")
        print(f"Total records processed: {len(rows_to_process)}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed to process: {failed_count}")
        print(f"Records with reasoning: {reasoning_dataset['llm_reasoning'].notna().sum()}")
        print(f"Output saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving dataset to {output_filename}: {str(e)}")
        sys.exit(1)
    
    return reasoning_dataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process progress notes with OpenAI reasoning and create a reasoning dataset.'
    )
    parser.add_argument(
        '--num-rows', 
        type=int,
        default=None,
        help='Number of rows to process. If not specified, processes all rows.'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        default='gpt-o3-mini',
        help='OpenAI model to use'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/datasets',
        help='Directory to save output files'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    openai_config = OpenAIConfig(model_name=args.model_name)
    reasoning_dataset = process_notes_with_reasoning(
        openai_config=openai_config,
        output_dir=args.output_dir,
        num_rows=args.num_rows
    )

