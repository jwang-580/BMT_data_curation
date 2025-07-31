#!/usr/bin/env python3
"""
Research data fields Script - Extract research data fields from medical notes using local LLMs via Ollama

This script extracts research data fields from daily progress notes (D0 to D+5) and discharge summary using different local LLMs with structured output.
The research data fields include patient demographics, disease information, and transplant details.

"""

import pandas as pd
import json
import time
from typing import Dict, List, Optional, Any, Literal
import argparse
from pathlib import Path
from ollama import chat
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# Research fields to extract from medical notes
RESEARCH_FIELDS = ['crs_y_n', 'fever_onset_date', 'last_fever_date', 'max_temp', 'hypotension_y_n',
                   'pressor_use_num', 'hypoxia_y_n', 'high_flow_o2_y_n', 'bipap_or_intubation_y_n', 
                   'neurotox_y_n', 'toci_y_n', 'toci_start_date', 'toci_stop_date', 'total_dose_toci']


class ResearchExtraction(BaseModel):
    """Pydantic model for structured research information extraction"""
    
    MRN: Optional[str] = Field(None, description="Medical Record Number - unique patient identifier")
    crs_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Cytokine Release Syndrome occured (Y/N)")
    fever_onset_date: Optional[str] = Field(None, description="Date of fever onset (YYYY-MM-DD format)")
    last_fever_date: Optional[str] = Field(None, description="Date of last fever (YYYY-MM-DD format)")
    max_temp: Optional[float] = Field(None, description="Maximum temperature in Celsius", ge=35.0, le=45.0)
    hypotension_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Hypotension occured (Y/N)")
    pressor_use_num: Optional[int] = Field(None, description="Number of vasopressors used", ge=0, le=10)
    hypoxia_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Hypoxia requiring supplemental oxygen (Y/N)")
    high_flow_o2_y_n: Optional[Literal["Y", "N"]] = Field(None, description="High flow oxygen requirement (Y/N)")
    bipap_or_intubation_y_n: Optional[Literal["Y", "N"]] = Field(None, description="BIPAP or intubation required (Y/N)")
    neurotox_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Neurotoxicity occured (Y/N)")
    toci_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Tocilizumab administered (Y/N)")
    toci_start_date: Optional[str] = Field(None, description="Tocilizumab start date (YYYY-MM-DD format)")
    toci_stop_date: Optional[str] = Field(None, description="Tocilizumab stop date (YYYY-MM-DD format)")
    total_dose_toci: Optional[float] = Field(None, description="Total doses of tocilizumab administered", ge=0)

class OllamaExtractor:
    """Extract medical information using Ollama local LLMs with structured output"""
    
    def __init__(self):
        pass
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            # First try to list models (lighter operation)
            import ollama
            models_response = ollama.list()
            print(f"Ollama connection successful - found models endpoint")
            return True
        except Exception as e:
            print(f"Failed to connect to Ollama: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            import ollama
            models_response = ollama.list()
            
            # Debug: Print the actual structure
            print(f"Debug - Ollama list response: {models_response}")
            
            # Handle different possible response structures
            if isinstance(models_response, dict):
                if 'models' in models_response:
                    models_list = models_response['models']
                else:
                    models_list = models_response
            else:
                models_list = models_response
            
            # Extract model names with multiple fallbacks
            model_names = []
            for model in models_list:
                if isinstance(model, dict):
                    # Try different possible keys
                    name = (model.get('name') or 
                           model.get('model') or 
                           model.get('id') or 
                           str(model))
                    model_names.append(name)
                else:
                    # If it's not a dict, use string representation
                    model_names.append(str(model))
            
            print(f"Extracted model names: {model_names}")
            return model_names
            
        except Exception as e:
            print(f"Failed to list models: {e}")
            # Try alternative approach with direct ollama command
            try:
                import subprocess
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    models = [line.split()[0] for line in lines if line.strip()]
                    print(f"Models from CLI: {models}")
                    return models
            except Exception as cli_e:
                print(f"CLI fallback also failed: {cli_e}")
            
            return []
    
    def extract_with_model(self, model_name: str, note_text: str, mrn: str, missing_fields: List[str] = None, day_info: str = "", accumulated_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract research fields using a specific model with structured output"""
        
        # Create extraction prompt (targeted if missing_fields provided)
        prompt = self._create_extraction_prompt(note_text, missing_fields, day_info, accumulated_data)
        
        try:
            # Make structured request to Ollama
            response = chat(
                model=model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a medical information extraction expert. Extract the requested baseline information from medical notes accurately and completely.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                format=ResearchExtraction.model_json_schema(),
                options={
                    'temperature': 0.1,  # Low temperature for consistency
                    'top_p': 0.9,
                }
            )
            
            # Parse structured response
            extracted_data = ResearchExtraction.model_validate_json(response.message.content)
            
            # Convert to dict and ensure MRN is set
            result_dict = extracted_data.model_dump()
            result_dict['MRN'] = mrn  # Ensure MRN is always set correctly
            result_dict['model_used'] = model_name
            
            # Check if any meaningful research data was actually extracted
            research_fields_extracted = sum(1 for field in RESEARCH_FIELDS 
                                           if field != 'MRN' and 
                                           result_dict.get(field) is not None and 
                                           result_dict.get(field) != '')
            
            # Only mark as successful if we extracted at least one research field
            result_dict['extraction_success'] = research_fields_extracted > 0
            result_dict['fields_extracted_count'] = research_fields_extracted
            
            if research_fields_extracted == 0:
                print(f"  Warning: Valid JSON returned but no research fields extracted for MRN {mrn}")
            
            return result_dict
                
        except Exception as e:
            print(f"Extraction failed for model {model_name}, MRN {mrn}: {e}")
            
            # Add more detailed debugging
            if hasattr(e, 'response'):
                print(f"  Response status: {getattr(e.response, 'status_code', 'Unknown')}")
            
            # Try to get the raw response if available
            try:
                if 'response' in locals():
                    print(f"  Raw response content: {response.message.content[:200]}...")
            except:
                pass
                
            return self._create_empty_result(mrn, model_name, success=False)
    
    def _create_extraction_prompt(self, note_text: str, missing_fields: List[str] = None, day_info: str = "", accumulated_data: Dict[str, Any] = None) -> str:
        """Create a structured prompt for medical information extraction"""
        
        # Field descriptions for research data
        field_descriptions = {
            'MRN': 'Medical Record Number - unique patient identifier',
            'crs_y_n': 'Whether Cytokine Release Syndrome occured or not(Y/N); if a fever is present, then CRS is likely to be present even though not mentioned in the note',
            'fever_onset_date': 'Date of fever onset (YYYY-MM-DD format)',
            'last_fever_date': 'Date of last fever (YYYY-MM-DD format)',
            'max_temp': 'Maximum temperature in Celsius',
            'hypotension_y_n': 'Hypotension occured, requiring fluid bolus or vasopressors (Y/N)',
            'pressor_use_num': 'If hypotension needing pressors, the number of vasopressors used',
            'hypoxia_y_n': 'Hypoxia requiring supplemental oxygen (Y/N)',
            'high_flow_o2_y_n': 'High flow oxygen requirement (Y/N)',
            'bipap_or_intubation_y_n': 'BIPAP or intubation required (Y/N)',
            'neurotox_y_n': 'Neurotoxicity occured (Y/N), any mention of altered mental status, confusion, or other neurological symptoms',
            'toci_y_n': 'Tocilizumab administered (Y/N)',
            'toci_start_date': 'Tocilizumab start date (YYYY-MM-DD format)',
            'toci_stop_date': 'Tocilizumab stop date (YYYY-MM-DD format)',
            'total_dose_toci': 'Total doses of tocilizumab administered, typically 1 or more doses'
        }
        
        # Build accumulated data section if available
        accumulated_info = ""
        if accumulated_data:
            accumulated_info = "CURRENT ACCUMULATED RESEARCH DATA (from previous days):\n"
            for field in RESEARCH_FIELDS:
                if field != 'MRN':
                    value = accumulated_data.get(field)
                    accumulated_info += f"- {field}: {value}\n"
            accumulated_info += "\n"
        

        all_fields_desc = []
        for field in RESEARCH_FIELDS:
            if field in field_descriptions:
                all_fields_desc.append(f"- {field}: {field_descriptions[field]}")
            else:
                all_fields_desc.append(f"- {field}")
        
        if "Discharge Summary" in day_info:
             prompt = f"""
{accumulated_info}DISCHARGE SUMMARY - FINAL UPDATE:
{note_text}

Based on the current accumulated data from all progress notes and this discharge summary, update any research fields necessary:
{chr(10).join(all_fields_desc)}

IMPORTANT INSTRUCTIONS:
- Review the accumulated data from all previous progress notes AND this discharge summary
- Discharge summaries often contain final outcomes, complications, and complete medication histories, especially neurotoxicity and tocilizumab information
- Return the COMPLETE final research data based on all information
"""
        else:
             prompt = f"""
{accumulated_info}DAILY PROGRESS NOTE - {day_info}:
{note_text}

Based on the current accumulated data (if any) and this new daily note, update any research fields if necessary:
{chr(10).join(all_fields_desc)}

IMPORTANT INSTRUCTIONS:
- Review the accumulated data from previous days AND this new daily note
- Updates any fields if there is new information in the daily note
- For temperatures: keep the highest temperature seen across all days
- For Y/N fields: mark Y if the condition occurred at any point across all days
- For counts/doses: accumulate totals across all days
- For ongoing conditions: update with the latest status from today's note
- Return the COMPLETE updated research data (not just changes from today)
"""
        
        return prompt
    
    def _create_empty_result(self, mrn: str, model_name: str, success: bool = False) -> Dict[str, Any]:
        """Create empty result structure"""
        result = {field: None for field in RESEARCH_FIELDS}
        result['MRN'] = mrn
        result['model_used'] = model_name
        result['extraction_success'] = success
        result['fields_extracted_count'] = 0
        return result

def load_progress_notes_data(file_path: str) -> pd.DataFrame:
    """Load the progress notes data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} progress notes from {file_path}")
        return df
    except Exception as e:
        print(f"Failed to load progress notes data: {e}")
        raise


def load_dc_summary_data(file_path: str) -> pd.DataFrame:
    """Load the discharge summary notes data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} discharge summary notes from {file_path}")
        return df
    except Exception as e:
        print(f"Failed to load discharge summary data: {e}")
        raise



def select_encounters_to_process(notes_df: pd.DataFrame, 
                                limit: Optional[int] = None,
                                rows: Optional[List[int]] = None,
                                row_range: Optional[List[int]] = None,
                                mrns: Optional[List[str]] = None,
                                random_sample: Optional[int] = None) -> pd.DataFrame:
    """Select specific encounters (MRN+BMT_date combinations) to process, then return all notes for selected encounters"""
    
    # First, get unique encounters (MRN+BMT_date combinations)
    unique_encounters = notes_df[['MRN', 'bmt_date']].drop_duplicates().reset_index(drop=True)
    print(f"Total unique encounters available: {len(unique_encounters)}")
    
    # Start with all encounters
    selected_encounters = unique_encounters.copy()
    
    # Apply MRN filter first if specified
    if mrns:
        mrn_strings = [str(mrn) for mrn in mrns]
        selected_encounters = selected_encounters[selected_encounters['MRN'].astype(str).isin(mrn_strings)]
        print(f"Filtered to {len(selected_encounters)} encounters matching MRNs: {mrns}")
    
    # Apply specific encounter indices
    if rows:
        # Ensure row indices are within bounds
        valid_rows = [r for r in rows if 0 <= r < len(selected_encounters)]
        if len(valid_rows) != len(rows):
            invalid_rows = [r for r in rows if r not in valid_rows]
            print(f"Warning: Invalid encounter indices ignored: {invalid_rows}")
        selected_encounters = selected_encounters.iloc[valid_rows]
        print(f"Selected specific encounters: {valid_rows}, resulting in {len(selected_encounters)} encounters")
    
    # Apply encounter range
    elif row_range:
        start, end = row_range
        start = max(0, start)
        end = min(len(selected_encounters), end)
        selected_encounters = selected_encounters.iloc[start:end]
        print(f"Selected encounter range [{start}:{end}), resulting in {len(selected_encounters)} encounters")
    
    # Apply random sampling
    elif random_sample:
        if random_sample > len(selected_encounters):
            print(f"Warning: Random sample size {random_sample} larger than available encounters {len(selected_encounters)}")
            random_sample = len(selected_encounters)
        selected_encounters = selected_encounters.sample(n=random_sample, random_state=42)
        print(f"Random sample of {random_sample} encounters selected")
    
    # Apply simple limit (lowest priority)
    elif limit:
        selected_encounters = selected_encounters.head(limit)
        print(f"Limited to first {limit} encounters")
    
    # Now get ALL notes for the selected encounters (all hospital days for each encounter)
    if len(selected_encounters) == 0:
        print("No encounters selected")
        return pd.DataFrame()
    
    # Create filter conditions for selected encounters
    encounter_conditions = []
    for _, encounter in selected_encounters.iterrows():
        condition = (notes_df['MRN'] == encounter['MRN']) & (notes_df['bmt_date'] == encounter['bmt_date'])
        encounter_conditions.append(condition)
    
    # Combine all conditions with OR
    combined_condition = encounter_conditions[0]
    for condition in encounter_conditions[1:]:
        combined_condition = combined_condition | condition
    
    # Filter notes to get all days for selected encounters
    selected_notes = notes_df[combined_condition].copy()
    
    print(f"Selected {len(selected_encounters)} encounters, returning {len(selected_notes)} total notes (all hospital days)")
    
    # Show some statistics
    days_per_encounter = selected_notes.groupby(['MRN', 'bmt_date']).size()
    print(f"Average days per encounter: {days_per_encounter.mean():.1f}")
    print(f"Days per encounter range: {days_per_encounter.min()} to {days_per_encounter.max()}")
    
    return selected_notes


def group_notes_by_mrn_bmt_and_day(notes_df: pd.DataFrame) -> Dict[str, Dict]:
    """Group notes by MRN+BMT_date combination and day for sequential processing"""
    grouped_notes = {}
    
    # Get unique combinations of MRN and bmt_date
    unique_combinations = notes_df[['MRN', 'bmt_date']].drop_duplicates()
    
    print(f"Found {len(unique_combinations)} unique MRN+BMT_date combinations")
    print(f"Total MRNs: {notes_df['MRN'].nunique()}")
    
    # Check for MRNs with multiple BMT dates
    mrn_counts = unique_combinations['MRN'].value_counts()
    multiple_bmts = mrn_counts[mrn_counts > 1]
    if len(multiple_bmts) > 0:
        print(f"MRNs with multiple BMT encounters: {len(multiple_bmts)}")
        print(f"  Examples: {list(multiple_bmts.head(3).index)}")
    
    for _, row in unique_combinations.iterrows():
        mrn = row['MRN']
        bmt_date = row['bmt_date']
        
        # Create a unique key for this MRN+BMT_date combination
        encounter_key = f"{mrn}_{bmt_date}"
        
        # Filter notes for this specific MRN+BMT_date combination
        encounter_notes = notes_df[
            (notes_df['MRN'] == mrn) & 
            (notes_df['bmt_date'] == bmt_date)
        ].copy()
        
        if len(encounter_notes) == 0:
            continue
            
        # Sort by note_date to ensure chronological order
        encounter_notes = encounter_notes.sort_values('note_date')
        
        # Group by day (note_date)
        daily_notes = []
        for day, day_notes in encounter_notes.groupby('note_date'):
            daily_notes.append(day_notes)
        
        grouped_notes[encounter_key] = {
            'mrn': str(mrn),
            'bmt_date': str(bmt_date),
            'daily_notes': daily_notes
        }
    
    return grouped_notes


def find_matching_dc_note(mrn: str, bmt_date: str, dc_summary_df: pd.DataFrame) -> Optional[str]:
    """Find matching discharge summary note for given MRN and bmt_date"""
    if dc_summary_df is None or len(dc_summary_df) == 0:
        return None
    
    # Ensure bmt_date is a datetime object
    bmt_date_dt = pd.to_datetime(bmt_date)

    # Ensure admit_date is in datetime format
    dc_summary_df = dc_summary_df.copy()
    dc_summary_df['admit_date'] = pd.to_datetime(dc_summary_df['admit_date'])

    # Match on MRN and admit_date within 30 days before bmt_date
    matches = dc_summary_df[
        (dc_summary_df['MRN'].astype(str) == str(mrn)) &
        (dc_summary_df['admit_date'] < bmt_date_dt) &
        (dc_summary_df['admit_date'] >= bmt_date_dt - timedelta(days=30))
    ]

    
    if len(matches) > 0:
        return str(matches.iloc[0]['note'])
    
    # Fallback: try MRN only (get most recent)
    mrn_matches = dc_summary_df[dc_summary_df['MRN'].astype(str) == str(mrn)]
    if len(mrn_matches) > 0:
        # Sort by admit_date if possible, otherwise take first
        try:
            mrn_matches_sorted = mrn_matches.sort_values('admit_date', ascending=False)
            return str(mrn_matches_sorted.iloc[0]['note'])
        except:
            return str(mrn_matches.iloc[0]['note'])
    
    return None


def extract_research_info(notes_df: pd.DataFrame, models: List[str], 
                         output_dir: str = "extracted_results", 
                         dc_summary_df: Optional[pd.DataFrame] = None,
                         limit: Optional[int] = None,
                         rows: Optional[List[int]] = None,
                         row_range: Optional[List[int]] = None,
                         mrns: Optional[List[str]] = None,
                         random_sample: Optional[int] = None) -> None:
    """Extract research information using multiple models with sequential daily note processing"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Select encounters to process based on various criteria
    notes_to_process = select_encounters_to_process(
        notes_df, limit=limit, rows=rows, row_range=row_range, 
        mrns=mrns, random_sample=random_sample
    )
    
    if len(notes_to_process) == 0:
        print("No encounters selected for processing. Check your selection criteria.")
        return
    
    # Count unique encounters in the selected notes
    unique_encounters_count = notes_to_process[['MRN', 'bmt_date']].drop_duplicates().shape[0]
    print(f"Processing {unique_encounters_count} encounters ({len(notes_to_process)} total daily notes)")
    
    # Initialize extractor
    extractor = OllamaExtractor()
    
    # Check connection
    if not extractor.check_ollama_connection():
        raise ConnectionError("Cannot connect to Ollama. Please ensure it's running.")
    
    # Verify models are available
    available_models = extractor.list_available_models()
    print(f"Available models: {available_models}")
    
    # Only use the models specified in the input list
    for model in models:
        # Check if model exists (allow partial matching for versioned models)
        model_exists = any(model in available_model for available_model in available_models)
        if not model_exists:
            print(f"Model {model} not found in Ollama. Available models: {available_models}")
            print(f"Attempting to use model {model} anyway...")
        
        print(f"Starting extraction with model: {model}")
        
        # Process notes
        results = []
        
        extraction_stats = {'success': 0, 'failed': 0}
        
        # Group notes by MRN+BMT_date for sequential daily processing
        grouped_notes = group_notes_by_mrn_bmt_and_day(notes_to_process)
        
        for encounter_key, encounter_data in grouped_notes.items():
            mrn = encounter_data['mrn']
            bmt_date = encounter_data['bmt_date']
            daily_notes_list = encounter_data['daily_notes']
            
            print(f"Processing encounter {encounter_key} (MRN {mrn}, BMT {bmt_date}) with {model} ({len(daily_notes_list)} days)")
            
            # Initialize accumulated result for this encounter
            accumulated_result = {
                'MRN': mrn,
                'bmt_date': bmt_date,
                'encounter_key': encounter_key,
                'model_used': model,
                'extraction_success': False,
                'fields_extracted_count': 0
            }
            
            # Process each day's notes sequentially
            for day_idx, day_notes_df in enumerate(daily_notes_list):
                day_date = str(day_notes_df.iloc[0]['note_date'])
                day_note_text = " ".join([str(note) for note in day_notes_df['note']])
                day_info = f"Day {day_idx + 1} - {day_date}"
                
                print(f"  Day {day_idx + 1}: {day_date}")
                
                # Extract research data from this day's notes, passing accumulated data
                daily_extracted_data = extractor.extract_with_model(
                    model, 
                    day_note_text, 
                    mrn, 
                    day_info=day_info,
                    accumulated_data=accumulated_result if day_idx > 0 else None
                )
                
                # Update accumulated result with today's extraction
                accumulated_result = daily_extracted_data
                accumulated_result['day_processed'] = day_idx + 1
                accumulated_result['total_days_processed'] = len(daily_notes_list)
                
                print(f"    Updated: {accumulated_result.get('fields_extracted_count', 0)} total fields")
                
                # Add small delay between days
                time.sleep(0.3)
            
            # FINAL STEP: Process DC summary if available to update all fields
            if dc_summary_df is not None:
                # Use the bmt_date from the encounter data
                dc_note_text = find_matching_dc_note(mrn, bmt_date, dc_summary_df)
                
                if dc_note_text:
                    print(f"  Final step: Processing DC summary")
                    
                    # Extract research data from DC summary, updating ALL fields
                    dc_extracted_data = extractor.extract_with_model(
                        model, 
                        dc_note_text, 
                        mrn, 
                        day_info="Final - Discharge Summary",
                        accumulated_data=accumulated_result
                    )
                    
                    # Update with DC summary results
                    accumulated_result = dc_extracted_data
                    accumulated_result['dc_summary_processed'] = True
                    accumulated_result['dc_note_length'] = len(dc_note_text)
                    
                    print(f"    Final update: {accumulated_result.get('fields_extracted_count', 0)} total fields")
                else:
                    print(f"  No matching DC summary found for encounter {encounter_key}")
                    accumulated_result['dc_summary_processed'] = False
            else:
                accumulated_result['dc_summary_processed'] = False
            
            # Track final success/failure
            if accumulated_result.get('extraction_success', False):
                extraction_stats['success'] += 1
            else:
                extraction_stats['failed'] += 1
                print(f"  FAILED: No research data extracted for encounter {encounter_key}")
            
            results.append(accumulated_result)
            
            # Add small delay between MRNs
            time.sleep(0.5)
        
        # Save results for this model
        results_df = pd.DataFrame(results)
        output_file = output_path / f"research_extraction_{model.replace(':', '_')}.csv"
        results_df.to_csv(output_file, index=False)
        
        # Print detailed summary
        successful_extractions = results_df['extraction_success'].sum()
        failed_extractions = len(results_df) - successful_extractions
        
        print(f"\nModel {model} - Extraction Summary:")
        print(f"  Total records processed: {len(results_df)}")
        print(f"  Successful extractions: {successful_extractions} ({successful_extractions/len(results_df)*100:.1f}%)")
        print(f"  Failed extractions: {failed_extractions} ({failed_extractions/len(results_df)*100:.1f}%)")
        
        # Show daily processing statistics
        if 'total_days_processed' in results_df.columns:
            total_days = results_df['total_days_processed'].sum()
            avg_days = results_df['total_days_processed'].mean()
            print(f"  Total days processed: {total_days}")
            print(f"  Average days per record: {avg_days:.1f}")
            
            # Show range of days processed
            min_days = results_df['total_days_processed'].min()
            max_days = results_df['total_days_processed'].max()
            print(f"  Days per record range: {min_days} to {max_days}")
        
        # Show DC summary processing statistics
        if 'dc_summary_processed' in results_df.columns:
            dc_processed = results_df['dc_summary_processed'].sum()
            dc_available = len(results_df[results_df['dc_summary_processed'] == True])
            print(f"  Records with DC summary processed: {dc_processed} ({dc_processed/len(results_df)*100:.1f}%)")
            
            if dc_available > 0:
                dc_rows = results_df[results_df['dc_summary_processed'] == True]
                avg_dc_length = dc_rows['dc_note_length'].mean() if 'dc_note_length' in dc_rows.columns else 0
                print(f"  Average DC summary length: {avg_dc_length:.0f} characters")
        
        # Show field extraction statistics for successful extractions
        if successful_extractions > 0:
            successful_rows = results_df[results_df['extraction_success'] == True]
            avg_fields_extracted = successful_rows['fields_extracted_count'].mean()
            print(f"  Average fields extracted per successful record: {avg_fields_extracted:.1f}/{len(RESEARCH_FIELDS)}")
            
            # Show rows with zero fields extracted
            zero_fields = len(successful_rows[successful_rows['fields_extracted_count'] == 0])
            if zero_fields > 0:
                print(f"  WARNING: {zero_fields} records marked successful but extracted no fields")
        
        print(f"Results saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract information from medical notes using local LLMs")
    
    parser.add_argument("--input", "-i", default="data/progress_notes.csv",
                       help="Input CSV file with progress notes (default: data/progress_notes.csv)")
    parser.add_argument("--dc-summary", default="data/dc_summary_notes.csv",
                       help="Optional CSV file with discharge summary notes for final update")
    parser.add_argument("--output", "-o", default="results/extracted_results",
                       help="Output directory (default: results/extracted_results)")
    parser.add_argument("--models", "-m", nargs="+", 
                       default=["medgemma_27b_fp16", "gemma3:27b-it-fp16", "llama3.1:70b"],
                       help="Models to use for extraction")
    parser.add_argument("--limit", "-l", type=int, default=None,
                       help="Limit number of encounters to process (for testing)")
    parser.add_argument("--rows", "-r", nargs="+", type=int, default=None,
                       help="Specific encounter indices to process (0-based, e.g., --rows 0 5 10)")
    parser.add_argument("--row-range", nargs=2, type=int, metavar=("START", "END"), default=None,
                       help="Process encounters in range [START, END) (0-based, e.g., --row-range 10 20)")
    parser.add_argument("--mrns", nargs="+", default=None,
                       help="Specific MRNs to process (e.g., --mrns 907009329 907163230)")
    parser.add_argument("--random-sample", type=int, default=None,
                       help="Process random sample of N encounters")
    parser.add_argument("--analyze-results", action="store_true",
                       help="Analyze existing extraction results")
    
    args = parser.parse_args()
    
    try:
        if args.analyze_results:
            # Analyze existing results
            analyze_extraction_results(args.output)
        else:
            # Load primary progress notes data
            notes_df = load_progress_notes_data(args.input)
            
            # Load DC summary data if provided
            dc_summary_df = None
            if args.dc_summary:
                try:
                    dc_summary_df = load_dc_summary_data(args.dc_summary)
                    print(f"DC summary will be used for final field updates")
                except Exception as e:
                    print(f"Warning: Failed to load DC summary data: {e}")
                    print("Proceeding with progress notes only")
            else:
                print("No DC summary provided - using progress notes only")
            
            # Extract research information
            extract_research_info(
                notes_df, 
                args.models, 
                args.output,
                dc_summary_df=dc_summary_df,
                limit=args.limit,
                rows=args.rows,
                row_range=args.row_range,
                mrns=args.mrns,
                random_sample=args.random_sample
            )
        
    except Exception as e:
        print(f"Research extraction failed: {e}")
        raise


def analyze_extraction_results(output_dir: str = "results/extracted_results") -> None:
    """Analyze extraction results to understand success/failure patterns"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Output directory {output_dir} not found")
        return
    
    result_files = list(output_path.glob("research_extraction_*.csv"))
    
    if not result_files:
        print("No extraction result files found")
        return
    
    print("\n" + "="*80)
    print("EXTRACTION RESULTS ANALYSIS")
    print("="*80)
    
    for file in result_files:
        model_name = file.stem.replace("research_extraction_", "")
        df = pd.read_csv(file)
        
        print(f"\n{model_name}:")
        print(f"  Total records: {len(df)}")
        
        # Check extraction success
        if 'extraction_success' in df.columns:
            successful = df['extraction_success'].sum()
            print(f"  Successful extractions: {successful}/{len(df)} ({successful/len(df)*100:.1f}%)")
        
        # Count empty rows (only MRN + metadata)
        research_cols = [col for col in RESEARCH_FIELDS if col in df.columns and col != 'MRN']
        empty_rows = df[research_cols].isna().all(axis=1).sum()
        print(f"  Completely empty rows: {empty_rows}")
        
        # Count rows with some data
        partial_rows = 0
        if research_cols:
            for _, row in df.iterrows():
                non_null_count = sum(1 for col in research_cols if pd.notna(row[col]) and row[col] != '')
                if 0 < non_null_count < len(research_cols):
                    partial_rows += 1
        
        print(f"  Partially filled rows: {partial_rows}")
        print(f"  Fully populated rows: {len(df) - empty_rows - partial_rows}")
        
        # Show field-by-field extraction rates
        if research_cols:
            print("  Field extraction rates:")
            for field in research_cols:
                if field in df.columns:
                    extracted = df[field].notna().sum()
                    rate = extracted / len(df) * 100
                    print(f"    {field}: {extracted}/{len(df)} ({rate:.1f}%)")
        
        # Show two-step extraction statistics if available
        if 'secondary_extraction_used' in df.columns:
            secondary_used = df['secondary_extraction_used'].sum()
            if secondary_used > 0:
                print(f"  Records enhanced by DC summary: {secondary_used}")
                
                # Show which fields were most commonly filled by secondary extraction
                if 'fields_filled_by_secondary' in df.columns:
                    all_filled_fields = []
                    for fields_list in df['fields_filled_by_secondary'].dropna():
                        if isinstance(fields_list, str) and fields_list.startswith('['):
                            # Handle string representation of list
                            import ast
                            try:
                                fields = ast.literal_eval(fields_list)
                                all_filled_fields.extend(fields)
                            except:
                                pass
                        elif isinstance(fields_list, list):
                            all_filled_fields.extend(fields_list)
                    
                    if all_filled_fields:
                        from collections import Counter
                        field_counts = Counter(all_filled_fields)
                        print("    Most commonly filled fields by DC summary:")
                        for field, count in field_counts.most_common(5):
                            print(f"      {field}: {count} times")
        
        # Show some example failed extractions if any
        if 'extraction_success' in df.columns:
            failed_df = df[df['extraction_success'] == False]
            if len(failed_df) > 0:
                print(f"  Sample failed MRNs: {list(failed_df['MRN'].head(5))}")
                
        # Show successful extractions with zero fields
        if 'extraction_success' in df.columns and 'fields_extracted_count' in df.columns:
            zero_fields_df = df[(df['extraction_success'] == True) & (df['fields_extracted_count'] == 0)]
            if len(zero_fields_df) > 0:
                print(f"  Successful but empty extractions: {len(zero_fields_df)} records")
                print(f"    Sample MRNs: {list(zero_fields_df['MRN'].head(5))}")


if __name__ == "__main__":
    main()
