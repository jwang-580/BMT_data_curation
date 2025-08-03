#!/usr/bin/env python3
"""
Research data fields Script - Extract research data fields from medical notes using local LLMs via Ollama

This script uses a simple two-step approach:
STEP 1: Extract daily clinical findings from progress notes, plus neurotoxicity from discharge summary
STEP 2: Combine all daily findings + DC neurotoxicity into final research fields

Discharge summary is processed separately to extract only neurotoxicity for the whole hospitalization.
All medical logic and rules are handled by the LLM via prompts - no hard-coded restrictions.

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


class DailyFindings(BaseModel):
    """Pydantic model for daily clinical findings extraction"""
    
    day_date: Optional[str] = Field(None, description="Date of this daily note (YYYY-MM-DD format)")
    max_temp_today: Optional[float] = Field(None, description="Maximum temperature today in Fahrenheit", ge=95.0, le=110.0)
    fever_today: Optional[Literal["Y", "N"]] = Field(None, description="Whether fever (Tmax > 100.3F) occurred today")
    crs_signs_today: Optional[Literal["Y", "N"]] = Field(None, description="Whether cytokine release syndrome (CRS) occurred today (if fever occurred today, then CRS occurred today)")
    hypotension_today: Optional[Literal["Y", "N"]] = Field(None, description="Hypotension requiring fluids/pressors today")
    pressor_count_today: Optional[int] = Field(None, description="Number of vasopressors used today", ge=0, le=10)
    hypoxia_today: Optional[Literal["Y", "N"]] = Field(None, description="Hypoxia requiring oxygen today")
    high_flow_o2_today: Optional[Literal["Y", "N"]] = Field(None, description="High flow oxygen used today")
    bipap_intubation_today: Optional[Literal["Y", "N"]] = Field(None, description="BIPAP or intubation today")
    neurotox_today: Optional[Literal["Y", "N"]] = Field(None, description="Neurotoxicity symptoms (altered mental status, confusion, seizures, etc.) today")
    tocilizumab_today: Optional[Literal["Y", "N"]] = Field(None, description="Whether tocilizumab was used today to mitigate CRS")
    toci_doses_today: Optional[float] = Field(None, description="Number of tocilizumab doses given today", ge=0, le=3)
    tocilizumab_note: Optional[str] = Field(None, description="Note about tocilizumab use recently")


class DischargeSummaryFindings(BaseModel):
    """Pydantic model for discharge summary neurotoxicity extraction"""
    
    neurotox_hospitalization: Optional[Literal["Y", "N"]] = Field(None, description="Neurotoxicity (altered mental status, confusion, seizures, etc.) occurred during the whole hospitalization (Y/N)")


class ResearchExtraction(BaseModel):
    """Pydantic model for structured research information extraction"""
    
    MRN: Optional[str] = Field(None, description="Medical Record Number - unique patient identifier")
    crs_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Cytokine Release Syndrome occured (Y/N)")
    fever_onset_date: Optional[str] = Field(None, description="Date of fever onset (YYYY-MM-DD format) or None")
    last_fever_date: Optional[str] = Field(None, description="Date of last fever (YYYY-MM-DD format) or None")
    max_temp: Optional[float] = Field(None, description="Maximum temperature in Fahrenheit", ge=95.0, le=110.0)
    hypotension_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Hypotension occured (Y/N)")
    pressor_use_num: Optional[int] = Field(None, description="Number of vasopressors used", ge=0, le=10)
    hypoxia_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Hypoxia requiring supplemental oxygen (Y/N)")
    high_flow_o2_y_n: Optional[Literal["Y", "N"]] = Field(None, description="High flow oxygen requirement (Y/N)")
    bipap_or_intubation_y_n: Optional[Literal["Y", "N"]] = Field(None, description="BIPAP or intubation required (Y/N)")
    neurotox_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Neurotoxicity occured (Y/N)")
    toci_y_n: Optional[Literal["Y", "N"]] = Field(None, description="Tocilizumab administered (Y/N)")
    toci_start_date: Optional[str] = Field(None, description="Tocilizumab start date (YYYY-MM-DD format) or None")
    toci_stop_date: Optional[str] = Field(None, description="Tocilizumab stop date (YYYY-MM-DD format) or None")
    total_dose_toci: Optional[float] = Field(None, description="Total doses of tocilizumab administered", ge=0, le=3)

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
    
    def extract_daily_findings(self, model_name: str, note_text: str, day_date: str, day_info: str) -> Dict[str, Any]:
        """STEP 1: Extract simple daily findings from a single day's notes"""
        
        prompt = f"""
MEDICAL RECORDS FOR {day_info}:
{note_text}

WHAT YOU ARE DOING:
You are extracting clinical findings from today's medical notes only. 

YOUR TASK:
Extract only TODAY's clinical findings. Look for:
- Maximum temperature today (look for "PHYSICAL EXAM  Temp:")
- Whether there is fever today (temperature > 100.3F), Y/N
- Whether there is CRS today (if fever occurred today, then CRS occurred today), Y/N
- Hypotension requiring treatment today
- Number of vasopressors used today
- Hypoxia requiring oxygen today
- High flow oxygen or respiratory support today
- BIPAP or intubation today
- Neurotoxicity symptoms today (confusion, altered mental status, seizures)
- If tocilizumab is used today to mitigate CRS, Y/N
- Number of tocilizumab doses given today
- Note about tocilizumab use recently

"""
        
        try:
            # Make structured request to Ollama
            response = chat(
                model=model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a medical information extraction expert. Extract only the clinical findings from today\'s medical note accurately.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                format=DailyFindings.model_json_schema(),
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                }
            )
            
            # Parse structured response
            daily_findings = DailyFindings.model_validate_json(response.message.content)
            result_dict = daily_findings.model_dump()
            result_dict['day_date'] = day_date
            result_dict['extraction_success'] = True
            
            return result_dict
                
        except Exception as e:
            print(f"Daily extraction failed for {day_info}: {e}")
            return {
                'day_date': day_date,
                'extraction_success': False,
                'error': str(e)
            }
    
    def extract_dc_summary_neurotox(self, model_name: str, dc_note_text: str) -> Dict[str, Any]:
        """Extract neurotoxicity information from discharge summary for the whole hospitalization"""
        
        prompt = f"""
DISCHARGE SUMMARY:
{dc_note_text}

WHAT YOU ARE DOING:
You are reviewing a discharge summary to determine if neurotoxicity occurred during the ENTIRE hospitalization.

YOUR TASK:
Review the entire discharge summary and determine if the patient experienced neurotoxicity at ANY point during this hospitalization. Look for mentions of:
- Altered mental status
- Confusion or delirium
- Seizures
- Neurological symptoms
- Cognitive changes
- Encephalopathy
- Any mention of neurotoxicity

"""
        
        try:
            # Make structured request to Ollama
            response = chat(
                model=model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a medical expert reviewing discharge summaries for neurotoxicity during hospitalization.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                format=DischargeSummaryFindings.model_json_schema(),
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                }
            )
            
            # Parse structured response
            dc_findings = DischargeSummaryFindings.model_validate_json(response.message.content)
            result_dict = dc_findings.model_dump()
            result_dict['extraction_success'] = True
            
            return result_dict
                
        except Exception as e:
            print(f"DC summary neurotox extraction failed: {e}")
            return {
                'neurotox_hospitalization': None,
                'extraction_success': False,
                'error': str(e)
            }
    
    def combine_daily_findings(self, model_name: str, daily_findings_list: List[Dict], mrn: str, dc_neurotox_findings: Dict[str, Any] = None) -> Dict[str, Any]:
        """STEP 2: Simply combine all daily findings into final research fields"""
        
        # Format daily findings for the prompt
        daily_summary = ""
        for i, findings in enumerate(daily_findings_list):
            if findings.get('extraction_success', False):
                daily_summary += f"\nDay {i+1} ({findings.get('day_date', 'Unknown')}):\n"
                for key, value in findings.items():
                    if key not in ['day_date', 'extraction_success', 'error'] and value is not None:
                        daily_summary += f"  {key}: {value}\n"
        
        # Add DC neurotoxicity findings if available
        dc_neurotox_section = ""
        if dc_neurotox_findings and dc_neurotox_findings.get('extraction_success', False):
            dc_neurotox_section = f"\nDISCHARGE SUMMARY NEUROTOXICITY (whole hospitalization):\n"
            neurotox_value = dc_neurotox_findings.get('neurotox_hospitalization')
            if neurotox_value is not None:
                dc_neurotox_section += f"  neurotoxicity_during_hospitalization: {neurotox_value}\n"
        
            prompt = f"""
DAILY FINDINGS ACROSS HOSPITALIZATION:
{daily_summary}
{dc_neurotox_section}

WHAT YOU ARE DOING:
You are combining daily clinical findings into final research data fields for a stem cell transplant study investigating cytokine release syndrome (CRS) and use to tocilizumab.

YOUR TASK:
Review all the daily findings above and combine them into final research fields. You must provide values for ALL research fields listed.

UNDERSTANDING THE DAILY FIELDS:
Each day shows findings for THAT DAY ONLY:
- max_temp_today: Highest temperature recorded on this specific day (Fahrenheit)
- fever_today: Was there fever (>100.3F) on this specific day? (Y/N)
- crs_signs_today: Were there is CRS on this specific day? (Y/N)
- hypotension_today: Was there hypotension requiring treatment on this specific day? (Y/N)
- pressor_count_today: Number of vasopressors used on this specific day (0, 1, 2, etc.)
- hypoxia_today: Was there hypoxia requiring oxygen on this specific day? (Y/N)
- high_flow_o2_today: Was high flow oxygen used on this specific day? (Y/N)
- bipap_intubation_today: Was BIPAP or intubation used on this specific day? (Y/N)
- neurotox_today: Were there neurotoxicity symptoms on this specific day? (Y/N)
- tocilizumab_today: Was tocilizumab given on this specific day? (Y/N)
- toci_doses_today: Number of tocilizumab doses given on this specific day (0, 1, 2, etc.)
- tocilizumab_note: Note about recent tocilizumab use. Sometimes tocilizumab use is mentioned in the note of next day. 

HOW TO COMBINE ACROSS DAYS:
- Y/N fields: If ANY day shows "Y", then final result = "Y". Only if all days show "N", then "N"
- Dates: Find earliest date for onset, latest date for end
- Numbers: Take highest value for maximums, add up for totals

REQUIRED RESEARCH FIELDS:
- crs_y_n: Overall CRS occurrence (Y/N) - "Y" if any day shows crs_signs_today="Y" OR fever_today="Y"
- fever_onset_date: Date of first fever (YYYY-MM-DD or null) - earliest day with fever_today="Y"
- last_fever_date: Date of last fever (YYYY-MM-DD or null) - latest day with fever_today="Y"
- max_temp: Highest temperature across all days - maximum of all max_temp_today values
- hypotension_y_n: Any hypotension occurrence (Y/N) - "Y" if any day shows hypotension_today="Y"
- pressor_use_num: Maximum number of pressors used - highest pressor_count_today value
- hypoxia_y_n: Any hypoxia occurrence (Y/N) - "Y" if any day shows hypoxia_today="Y"
- high_flow_o2_y_n: Any high flow oxygen use (Y/N) - "Y" if any day shows high_flow_o2_today="Y"
- bipap_or_intubation_y_n: Any BIPAP/intubation use (Y/N) - "Y" if any day shows bipap_intubation_today="Y"
- neurotox_y_n: Any neurotoxicity occurrence (Y/N) - USE DISCHARGE SUMMARY if available, otherwise "Y" if any day shows neurotox_today="Y"
- toci_y_n: Any tocilizumab administration (Y/N) - "Y" if any day shows tocilizumab_today="Y"
- toci_start_date: First tocilizumab date (YYYY-MM-DD or null) - earliest day with tocilizumab_today="Y"
- toci_stop_date: Last tocilizumab date (YYYY-MM-DD or null) - latest day with tocilizumab_today="Y"
- total_dose_toci: Total tocilizumab doses given - sum of all toci_doses_today values

EXAMPLES OF CORRECT LOGIC:
- If Day 1: crs_signs_today = "N", Day 2: crs_signs_today = "Y" → crs_y_n = "Y"
- If Day 1: fever_today = "Y", Day 2: fever_today = "N" → fever_onset_date = Day 1 date, last_fever_date = Day 1 date
- If Day 1: max_temp = 101.0, Day 2: max_temp = 103.5 → max_temp = 103.5
- If Day 1: toci_doses_today = 1.0, Day 2: toci_doses_today = 2.0 → total_dose_toci = 3.0

RETURN: Complete final research data with ALL required fields using the combination rules above.
"""
        
        try:
            # Make structured request to Ollama
            response = chat(
                model=model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a medical researcher combining daily findings into final research data. Apply proper medical logic to aggregate findings across days.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                format=ResearchExtraction.model_json_schema(),
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                }
            )
            
            # Parse structured response
            final_research = ResearchExtraction.model_validate_json(response.message.content)
            result_dict = final_research.model_dump()
            result_dict['MRN'] = mrn
            result_dict['model_used'] = model_name
            result_dict['extraction_success'] = True
            
            # Count extracted fields
            fields_extracted = sum(1 for field in RESEARCH_FIELDS 
                                 if field != 'MRN' and 
                                 result_dict.get(field) is not None and 
                                 result_dict.get(field) != '')
            result_dict['fields_extracted_count'] = fields_extracted
            
            return result_dict
                
        except Exception as e:
            print(f"Combination failed for MRN {mrn}: {e}")
            return self._create_empty_result(mrn, model_name, success=False)


    

    
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
    
    
    for model in models:
        
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
            
            # STEP 1: Extract daily findings for each day + DC summary
            daily_findings_list = []
            print(f"  STEP 1: Extracting daily findings...")
            
            # Process each daily progress note
            for day_idx, day_notes_df in enumerate(daily_notes_list):
                note_date_offset = int(day_notes_df.iloc[0]['note_date'])
                day_note_text = " ".join([str(note) for note in day_notes_df['note']])
                
                # Calculate actual date: BMT date + note_date offset
                bmt_date_dt = pd.to_datetime(bmt_date)
                actual_date = bmt_date_dt + pd.Timedelta(days=note_date_offset)
                actual_date_str = actual_date.strftime('%Y-%m-%d')
                
                day_info = f"Day {day_idx + 1} - {actual_date_str}"
                
                print(f"    Day {day_idx + 1}: Extracting findings from {actual_date_str} (BMT+{note_date_offset})")
                
                # Extract daily findings for this day only
                daily_findings = extractor.extract_daily_findings(
                    model, 
                    day_note_text, 
                    actual_date_str,
                    day_info
                )
                
                daily_findings_list.append(daily_findings)
                
                if daily_findings.get('extraction_success', False):
                    print(f"      ✓ Extracted daily findings")
                    # Print detailed findings for debugging
                    print(f"      📋 Daily findings (day_date: {daily_findings.get('day_date', 'N/A')}):")
                    for key, value in daily_findings.items():
                        if key not in ['day_date', 'extraction_success', 'error'] and value is not None:
                            print(f"        {key}: {value}")
                else:
                    print(f"      ✗ Failed to extract daily findings")
                    if 'error' in daily_findings:
                        print(f"        Error: {daily_findings['error']}")
                
                # Add small delay between days
                time.sleep(0.3)
            
            # Process DC summary separately for neurotoxicity only
            dc_neurotox_findings = None
            if dc_summary_df is not None:
                dc_note_text = find_matching_dc_note(mrn, bmt_date, dc_summary_df)
                if dc_note_text:
                    print(f"    Discharge Summary: Extracting neurotoxicity for whole hospitalization")
                    
                    # Extract only neurotoxicity from DC summary for whole hospitalization
                    dc_neurotox_findings = extractor.extract_dc_summary_neurotox(
                        model, 
                        dc_note_text
                    )
                    
                    if dc_neurotox_findings.get('extraction_success', False):
                        print(f"      ✓ Extracted DC neurotoxicity findings")
                        # Print detailed DC findings for debugging
                        print(f"      📋 DC neurotoxicity findings:")
                        for key, value in dc_neurotox_findings.items():
                            if key not in ['extraction_success', 'error'] and value is not None:
                                print(f"        {key}: {value}")
                else:
                        print(f"      ✗ Failed to extract DC neurotoxicity findings")
                        if 'error' in dc_neurotox_findings:
                            print(f"        Error: {dc_neurotox_findings['error']}")
            else:
                    print(f"    No DC summary found")
            
            # STEP 2: Simply combine all daily findings into final research fields
            print(f"  STEP 2: Combining all findings into final research data...")
            
            final_result = extractor.combine_daily_findings(
                model,
                daily_findings_list,
                mrn,
                dc_neurotox_findings
            )
            
            # Print final combined results for debugging
            if final_result.get('extraction_success', False):
                print(f"      ✓ Combined into final research fields")
                print(f"      🎯 Final research data (ALL fields):")
                for field in RESEARCH_FIELDS:
                    if field != 'MRN':
                        value = final_result.get(field)
                        # Show ALL fields, including null values, for debugging
                        print(f"        {field}: {value}")
            else:
                print(f"      ✗ Failed to combine findings")
                if 'error' in final_result:
                    print(f"        Error: {final_result['error']}")
            
            # Add metadata
            final_result['bmt_date'] = bmt_date
            final_result['encounter_key'] = encounter_key
            final_result['total_days_processed'] = len(daily_notes_list)
            final_result['daily_extractions_successful'] = sum(1 for df in daily_findings_list if df.get('extraction_success', False))
            final_result['dc_summary_processed'] = dc_neurotox_findings is not None and dc_neurotox_findings.get('extraction_success', False)
            final_result['dc_neurotox_extracted'] = dc_neurotox_findings.get('neurotox_hospitalization') if dc_neurotox_findings else None
            final_result['total_findings_extracted'] = len(daily_findings_list)
            
            # Track final success/failure
            if final_result.get('extraction_success', False):
                extraction_stats['success'] += 1
                print(f"  ✓ SUCCESS: Extracted {final_result.get('fields_extracted_count', 0)} research fields")
            else:
                extraction_stats['failed'] += 1
                print(f"  ✗ FAILED: No research data extracted for encounter {encounter_key}")
            
            results.append(final_result)
            
            # Add small delay between encounters
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
        
        # Show two-step processing statistics
        if 'total_days_processed' in results_df.columns:
            total_days = results_df['total_days_processed'].sum()
            avg_days = results_df['total_days_processed'].mean()
            print(f"  Total days processed: {total_days}")
            print(f"  Average days per encounter: {avg_days:.1f}")
            
            # Show range of days processed
            min_days = results_df['total_days_processed'].min()
            max_days = results_df['total_days_processed'].max()
            print(f"  Days per encounter range: {min_days} to {max_days}")
        
        # Show Step 1 (daily extraction) success rates
        if 'daily_extractions_successful' in results_df.columns and 'total_findings_extracted' in results_df.columns:
            total_step1_success = results_df['daily_extractions_successful'].sum()
            total_step1_attempted = results_df['total_findings_extracted'].sum()
            step1_success_rate = total_step1_success / total_step1_attempted * 100 if total_step1_attempted > 0 else 0
            print(f"  Step 1 (daily extraction) success rate: {total_step1_success}/{total_step1_attempted} ({step1_success_rate:.1f}%)")
        
        # Show DC summary processing statistics
        if 'dc_summary_processed' in results_df.columns:
            dc_processed = results_df['dc_summary_processed'].sum()
            print(f"  Records with DC summary neurotoxicity processed: {dc_processed} ({dc_processed/len(results_df)*100:.1f}%)")
            
            if 'dc_neurotox_extracted' in results_df.columns:
                neurotox_y_count = (results_df['dc_neurotox_extracted'] == 'Y').sum()
                neurotox_n_count = (results_df['dc_neurotox_extracted'] == 'N').sum()
                if neurotox_y_count > 0 or neurotox_n_count > 0:
                    print(f"  DC neurotoxicity findings: Y={neurotox_y_count}, N={neurotox_n_count}")
        
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
                    print(f"DC summary will be used for neurotoxicity extraction")
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
