#!/usr/bin/env python3
"""
Baseline Extract Script - Extract baseline fields from medical notes using local LLMs via Ollama

This script extracts baseline information from H&P notes and discharge summary notes using different local LLMs with structured output.
The baseline fields include patient demographics, disease information, and transplant details.

"""

import pandas as pd
import json
import time
from typing import Dict, List, Optional, Any, Literal
import argparse
from pathlib import Path
from ollama import chat
from pydantic import BaseModel, Field

# Baseline fields to extract from medical notes
BASELINE_FIELDS = [
    'MRN', 'BMT_date', 'Disease_x', 'age', 'Gn', 'KPS', 'CMV', 'aborh', 
    'Tx_Type', 'HLA', 'donabo', 'doncmv', 'Source_x', 'Prep', 'AB', 'gvhdpr'
]


class BaselineExtraction(BaseModel):
    """Pydantic model for structured baseline information extraction"""
    
    MRN: Optional[str] = Field(None, description="Medical Record Number - unique patient identifier")
    BMT_date: Optional[str] = Field(None, description="Bone Marrow Transplant date (format: YYYY-MM-DD)")
    Disease_x: Optional[Literal['MDS-MPN', 'AML', 'NHL', 'HOD', 'CLL', 'AcL', 'MPN', 'ALL', 'MDS', 'PLL', 'MM', 'CML']] = Field(None, description="Primary disease/diagnosis")
    age: Optional[int] = Field(None, description="Patient age in years", ge=0, le=120)
    Gn: Optional[Literal["M", "F"]] = Field(None, description="Gender (M/F)")
    KPS: Optional[int] = Field(None, description="Karnofsky Performance Score", ge=0, le=100)
    CMV: Optional[Literal["pos", "neg"]] = Field(None, description="CMV status (pos/neg)")
    aborh: Optional[str] = Field(None, description="ABO/Rh blood type (e.g., O+, A-, B+)")
    Tx_Type: Optional[Literal["MUD", "REL", "MMUD"]] = Field(None, description="Transplant donor type")
    HLA: Optional[str] = Field(None, description="HLA matching status out of 8 points(e.g., 8/8, 7/8)")
    donabo: Optional[str] = Field(None, description="Donor ABO/Rh blood type (e.g., O+, A-, B+)")
    doncmv: Optional[Literal["pos", "neg"]] = Field(None, description="Donor CMV status (pos/neg)")
    Source_x: Optional[Literal["PB", "BM"]] = Field(None, description="Stem cell source")
    Prep: Optional[Literal["FluCyTBI", "FluMel", "FluBu", "FluBuThio", "FluMelTBI", "FluTBI", "FluMelThio", "FluCyTBIATG", "FluMelMcrvPlac", "FluBuThioMcrvPlac", "FluTreo"]] = Field(None, description="Conditioning/preparative regimen")
    AB: Optional[Literal["ric", "abl"]] = Field(None, description="Conditioning regimen intensity type, reduced intensity (ric) or ablation (abl)")
    gvhdpr: Optional[Literal["FK / MMF / Cy", "cyclophosphamide", "siro / MMF / Cy", "FK / MMF / Cy 100", "FK / MMF / Cy / siro"]] = Field(None, description="GVHD prophylaxis regimen")

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
    
    def extract_with_model(self, model_name: str, note_text: str, mrn: str, missing_fields: List[str] = None) -> Dict[str, Any]:
        """Extract baseline fields using a specific model with structured output"""
        
        # Create extraction prompt (targeted if missing_fields provided)
        prompt = self._create_extraction_prompt(note_text, missing_fields)
        
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
                format=BaselineExtraction.model_json_schema(),
                options={
                    'temperature': 0.1,  # Low temperature for consistency
                    'top_p': 0.9,
                }
            )
            
            # Parse structured response
            extracted_data = BaselineExtraction.model_validate_json(response.message.content)
            
            # Convert to dict and ensure MRN is set
            result_dict = extracted_data.model_dump()
            result_dict['MRN'] = mrn  # Ensure MRN is always set correctly
            result_dict['model_used'] = model_name
            
            # Check if any meaningful baseline data was actually extracted
            baseline_fields_extracted = sum(1 for field in BASELINE_FIELDS 
                                           if field != 'MRN' and 
                                           result_dict.get(field) is not None and 
                                           result_dict.get(field) != '')
            
            # Only mark as successful if we extracted at least one baseline field
            result_dict['extraction_success'] = baseline_fields_extracted > 0
            result_dict['fields_extracted_count'] = baseline_fields_extracted
            
            if baseline_fields_extracted == 0:
                print(f"  Warning: Valid JSON returned but no baseline fields extracted for MRN {mrn}")
            
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
    
    def _create_extraction_prompt(self, note_text: str, missing_fields: List[str] = None) -> str:
        """Create a structured prompt for medical information extraction"""
        
        if missing_fields:
            # Create targeted prompt for missing fields
            missing_fields_desc = []
            field_descriptions = {
                'MRN': 'Medical Record Number',
                'BMT_date': 'Bone Marrow Transplant date (YYYY-MM-DD format)',
                'Disease_x': 'Primary disease/diagnosis (must be one of the specific disease codes like AML-NPM1 mut, MDS-CMML, NHL-DLBCL, etc.), be as specific as possible',
                'age': 'Patient age in years',
                'Gn': 'Gender (M or F only)',
                'KPS': 'Karnofsky Performance Score (0-100)',
                'CMV': 'CMV status (pos or neg only)',
                'aborh': 'ABO/Rh blood type (e.g., O+, A-, B+), use + and - to indicate positive and negative, respectively',
                'Tx_Type': 'Transplant donor type (MUD, REL, or MMUD only)',
                'HLA': 'HLA matching status out of ** 8 points ** (e.g., 8/8, 7/8, 4/8 for haploidentical transplant) not */10 or */12',
                'donabo': 'Donor ABO/Rh blood type (e.g., O+, A-, B+)',
                'doncmv': 'Donor CMV status (pos or neg only)',
                'Source_x': 'Stem cell source (PB or BM only)',
                'Prep': 'Conditioning/preparative regimen (FluCyTBI, FluMel, FluBu, FluBuThio, FluMelTBI, FluTBI, FluMelThio, FluCyTBIATG, FluMelMcrvPlac, FluBuThioMcrvPlac, or FluTreo)',
                'AB': 'Conditioning regimen intensity type reduced intensivty (ric) or ablation (abl)',
                'gvhdpr': 'GVHD prophylaxis regimen (FK / MMF / Cy, cyclophosphamide, siro / MMF / Cy, FK / MMF / Cy 100, or FK / MMF / Cy / siro)'
            }
            
            for field in missing_fields:
                if field in field_descriptions:
                    missing_fields_desc.append(f"- {field}: {field_descriptions[field]}")
            
            prompt = f"""
This is a SECOND-PASS extraction to fill in missing information and recheck Disease_x from a discharge summary note.

FOCUS ONLY on finding the following MISSING fields from previous extraction as well as rechecking Disease_x:

DISCHARGE SUMMARY NOTE:
{note_text}

MISSING FIELDS TO EXTRACT:
{chr(10).join(missing_fields_desc)}

IMPORTANT: 
- Focus specifically on the missing fields listed above as well as Disease_x
- Use medical knowledge to infer information when not explicitly stated
- Discharge summaries often contain transplant outcomes and final diagnoses
- Return null only if absolutely no information can be found or inferred
"""
        else:
            # Standard full extraction prompt
            prompt = f"""
Extract the following baseline medical information from the provided medical note. 
Be precise and extract information that is explicitly stated or can be reasonably inferred from the note.

IMPORTANT: Do not return null values unless absolutely no information can be found or inferred.

MEDICAL NOTE:
{note_text}

Please extract the following information:
- MRN: Medical Record Number
- BMT_date: Bone Marrow Transplant date (YYYY-MM-DD format)
- Disease_x: Primary disease/diagnosis (must be one of the specific disease codes like AML-NPM1 mut, MDS-CMML, NHL-DLBCL, etc.), be as specific as possible.
- age: Patient age in years
- Gn: Gender (M or F only)
- KPS: Karnofsky Performance Score (0-100)
- CMV: CMV status (pos or neg only)
- aborh: ABO/Rh blood type (e.g., O+, A-, B+), use + and - to indicate positive and negative, respectively.
- Tx_Type: Transplant donor type (MUD, REL, or MMUD only)
- HLA: HLA matching status out of ** 8 points ** (e.g., 8/8, 7/8, 4/8 for haploidentical transplant) not */10 or */12.
- donabo: Donor ABO/Rh blood type (e.g., O+, A-, B+)
- doncmv: Donor CMV status (pos or neg only)
- Source_x: Stem cell source (PB or BM only)
- Prep: Conditioning/preparative regimen (FluCyTBI, FluMel, FluBu, FluBuThio, FluMelTBI, FluTBI, FluMelThio, FluCyTBIATG, FluMelMcrvPlac, FluBuThioMcrvPlac, or FluTreo)
- AB: Conditioning regimen intensity type reduced intensivty (ric) or ablation (abl)
- gvhdpr: GVHD prophylaxis regimen (FK / MMF / Cy, cyclophosphamide, siro / MMF / Cy, FK / MMF / Cy 100, or FK / MMF / Cy / siro)

If any information is not explicitly stated, use your medical knowledge to make reasonable inferences. For example:
- If you see "Flu/Cy/TBI" conditioning, infer AB as "ric" (reduced intensity)
- If patient mentions being a "brother" or "sister", infer Tx_Type as "REL" (related donor)
- Extract at least some information rather than returning all null values.
"""
        
        return prompt
    
    def _create_empty_result(self, mrn: str, model_name: str, success: bool = False) -> Dict[str, Any]:
        """Create empty result structure"""
        result = {field: None for field in BASELINE_FIELDS}
        result['MRN'] = mrn
        result['model_used'] = model_name
        result['extraction_success'] = success
        result['fields_extracted_count'] = 0
        return result


def identify_missing_fields(extraction_result: Dict[str, Any]) -> List[str]:
    """Identify which baseline fields are missing or empty from extraction result"""
    missing_fields = []
    
    for field in BASELINE_FIELDS:
        if field != 'MRN':  # MRN should always be present
            value = extraction_result.get(field)
            if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
                missing_fields.append(field)
    
    return missing_fields


def merge_extraction_results(primary_result: Dict[str, Any], secondary_result: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two extraction results, using secondary to fill gaps in primary"""
    merged_result = primary_result.copy()
    
    # Track which fields were filled by secondary extraction
    fields_filled = []
    
    for field in BASELINE_FIELDS:
        primary_value = primary_result.get(field)
        secondary_value = secondary_result.get(field)
        
        # Use secondary value if primary is missing/empty and secondary has data
        if (primary_value is None or primary_value == '' or 
            (isinstance(primary_value, str) and primary_value.strip() == '')):
            if (secondary_value is not None and secondary_value != '' and 
                not (isinstance(secondary_value, str) and secondary_value.strip() == '')):
                merged_result[field] = secondary_value
                fields_filled.append(field)
    
    # Update metadata
    merged_result['extraction_success'] = True  # Will be recalculated based on final fields
    merged_result['secondary_extraction_used'] = len(fields_filled) > 0
    merged_result['fields_filled_by_secondary'] = fields_filled
    
    # Recalculate fields extracted count
    baseline_fields_extracted = sum(1 for field in BASELINE_FIELDS 
                                   if field != 'MRN' and 
                                   merged_result.get(field) is not None and 
                                   merged_result.get(field) != '')
    
    merged_result['extraction_success'] = baseline_fields_extracted > 0
    merged_result['fields_extracted_count'] = baseline_fields_extracted
    
    return merged_result


def find_matching_dc_note(mrn: str, admit_date: str, dc_summary_df: pd.DataFrame) -> Optional[str]:
    """Find matching discharge summary note for given MRN and admit_date"""
    if dc_summary_df is None or len(dc_summary_df) == 0:
        return None
    
    # Try exact match on MRN and admit_date
    matches = dc_summary_df[
        (dc_summary_df['MRN'].astype(str) == str(mrn)) &
        (dc_summary_df['admit_date'].astype(str) == str(admit_date))
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


def load_notes_data(file_path: str) -> pd.DataFrame:
    """Load the H&P notes data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} notes from {file_path}")
        return df
    except Exception as e:
        print(f"Failed to load notes data: {e}")
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


def select_rows_to_process(notes_df: pd.DataFrame, 
                          limit: Optional[int] = None,
                          rows: Optional[List[int]] = None,
                          row_range: Optional[List[int]] = None,
                          mrns: Optional[List[str]] = None,
                          random_sample: Optional[int] = None) -> pd.DataFrame:
    """Select specific rows to process based on various criteria"""
    
    # Start with all rows
    selected_df = notes_df.copy()
    
    # Apply MRN filter first if specified
    if mrns:
        mrn_strings = [str(mrn) for mrn in mrns]
        selected_df = selected_df[selected_df['MRN'].astype(str).isin(mrn_strings)]
        print(f"Filtered to {len(selected_df)} rows matching MRNs: {mrns}")
    
    # Apply specific row indices
    if rows:
        # Ensure row indices are within bounds
        valid_rows = [r for r in rows if 0 <= r < len(selected_df)]
        if len(valid_rows) != len(rows):
            invalid_rows = [r for r in rows if r not in valid_rows]
            print(f"Warning: Invalid row indices ignored: {invalid_rows}")
        selected_df = selected_df.iloc[valid_rows]
        print(f"Selected specific rows: {valid_rows}, resulting in {len(selected_df)} rows")
    
    # Apply row range
    elif row_range:
        start, end = row_range
        start = max(0, start)
        end = min(len(selected_df), end)
        selected_df = selected_df.iloc[start:end]
        print(f"Selected row range [{start}:{end}), resulting in {len(selected_df)} rows")
    
    # Apply random sampling
    elif random_sample:
        if random_sample > len(selected_df):
            print(f"Warning: Random sample size {random_sample} larger than available rows {len(selected_df)}")
            random_sample = len(selected_df)
        selected_df = selected_df.sample(n=random_sample, random_state=42)
        print(f"Random sample of {random_sample} rows selected")
    
    # Apply simple limit (lowest priority)
    elif limit:
        selected_df = selected_df.head(limit)
        print(f"Limited to first {limit} rows")
    
    return selected_df


def extract_baseline_info(notes_df: pd.DataFrame, models: List[str], 
                         output_dir: str = "extracted_results", 
                         dc_summary_df: Optional[pd.DataFrame] = None,
                         limit: Optional[int] = None,
                         rows: Optional[List[int]] = None,
                         row_range: Optional[List[int]] = None,
                         mrns: Optional[List[str]] = None,
                         random_sample: Optional[int] = None) -> None:
    """Extract baseline information using multiple models"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Select rows to process based on various criteria
    notes_to_process = select_rows_to_process(
        notes_df, limit=limit, rows=rows, row_range=row_range, 
        mrns=mrns, random_sample=random_sample
    )
    
    if len(notes_to_process) == 0:
        print("No rows selected for processing. Check your selection criteria.")
        return
    
    print(f"Processing {len(notes_to_process)} encounters/rows")
    
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
        
        extraction_stats = {'success': 0, 'failed': 0, 'secondary_used': 0}
        
        for idx, row in notes_to_process.iterrows():
            mrn = str(row['MRN'])
            hp_note_text = str(row['note'])
            admit_date = str(row['admit_date'])
            
            print(f"Processing MRN {mrn} with {model} ({idx+1}/{len(notes_to_process)})")
            
            # STEP 1: Extract from H&P notes
            extracted_data = extractor.extract_with_model(model, hp_note_text, mrn)
            extracted_data['admit_date'] = admit_date
            extracted_data['note_length'] = len(hp_note_text)
            extracted_data['hp_extraction_success'] = extracted_data.get('extraction_success', False)
            
            print(f"  H&P extraction: {extracted_data.get('fields_extracted_count', 0)} fields extracted")
            
            # STEP 2: Check if we need secondary extraction from DC summary
            missing_fields = identify_missing_fields(extracted_data)
            
            if dc_summary_df is not None and len(missing_fields) > 0:
                dc_note_text = find_matching_dc_note(mrn, admit_date, dc_summary_df)
                
                if dc_note_text:
                    print(f"  Found DC summary, attempting secondary extraction for {len(missing_fields)} missing fields")
                    
                    # Extract only missing fields from DC summary
                    dc_extracted_data = extractor.extract_with_model(model, dc_note_text, mrn, missing_fields)
                    
                    # Merge results
                    merged_data = merge_extraction_results(extracted_data, dc_extracted_data)
                    
                    # Add DC summary metadata
                    merged_data['dc_note_available'] = True
                    merged_data['dc_note_length'] = len(dc_note_text)
                    merged_data['dc_extraction_success'] = dc_extracted_data.get('extraction_success', False)
                    
                    if merged_data.get('secondary_extraction_used', False):
                        extraction_stats['secondary_used'] += 1
                        fields_filled = merged_data.get('fields_filled_by_secondary', [])
                        print(f"  DC summary filled {len(fields_filled)} additional fields: {fields_filled}")
                    
                    extracted_data = merged_data
                else:
                    print(f"  No matching DC summary found for MRN {mrn}")
                    extracted_data['dc_note_available'] = False
                    extracted_data['secondary_extraction_used'] = False
            else:
                extracted_data['dc_note_available'] = dc_summary_df is not None
                extracted_data['secondary_extraction_used'] = False
            
            # Track final success/failure
            if extracted_data.get('extraction_success', False):
                extraction_stats['success'] += 1
                fields_extracted = extracted_data.get('fields_extracted_count', 0)
                if fields_extracted == 0:
                    print(f"  WARNING: Extraction marked successful but no fields extracted for MRN {mrn}")
            else:
                extraction_stats['failed'] += 1
                print(f"  FAILED: No data extracted for MRN {mrn}")
            
            results.append(extracted_data)
            
            # Add small delay to avoid overwhelming the system
            time.sleep(0.5)
        
        # Save results for this model
        results_df = pd.DataFrame(results)
        output_file = output_path / f"baseline_extraction_{model.replace(':', '_')}.csv"
        results_df.to_csv(output_file, index=False)
        
        # Print detailed summary
        successful_extractions = results_df['extraction_success'].sum()
        failed_extractions = len(results_df) - successful_extractions
        secondary_used = extraction_stats['secondary_used']
        
        print(f"\nModel {model} - Extraction Summary:")
        print(f"  Total records processed: {len(results_df)}")
        print(f"  Successful extractions: {successful_extractions} ({successful_extractions/len(results_df)*100:.1f}%)")
        print(f"  Failed extractions: {failed_extractions} ({failed_extractions/len(results_df)*100:.1f}%)")
        
        # Show secondary extraction statistics
        if dc_summary_df is not None:
            dc_available = results_df['dc_note_available'].sum() if 'dc_note_available' in results_df.columns else 0
            print(f"  DC summary notes available: {dc_available}")
            print(f"  Records enhanced by DC summary: {secondary_used} ({secondary_used/len(results_df)*100:.1f}%)")
        
        # Show field extraction statistics for successful extractions
        if successful_extractions > 0:
            successful_rows = results_df[results_df['extraction_success'] == True]
            avg_fields_extracted = successful_rows['fields_extracted_count'].mean()
            print(f"  Average fields extracted per successful record: {avg_fields_extracted:.1f}/{len(BASELINE_FIELDS)}")
            
            # Show H&P vs final success rates
            if 'hp_extraction_success' in results_df.columns:
                hp_success = results_df['hp_extraction_success'].sum()
                improvement = successful_extractions - hp_success
                if improvement > 0:
                    print(f"  Improvement from DC summary: +{improvement} successful extractions")
            
            # Show rows with zero fields extracted
            zero_fields = len(successful_rows[successful_rows['fields_extracted_count'] == 0])
            if zero_fields > 0:
                print(f"  WARNING: {zero_fields} records marked successful but extracted no fields")
        
        print(f"Results saved to: {output_file}")





def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract baseline information from medical notes using local LLMs")
    
    parser.add_argument("--input", "-i", default="data/h_p_notes.csv",
                       help="Input CSV file with H&P notes (default: data/h_p_notes.csv)")
    parser.add_argument("--dc-summary", default="data/dc_summary_notes.csv",
                       help="Optional CSV file with discharge summary notes for secondary extraction")
    parser.add_argument("--output", "-o", default="results/extracted_results",
                       help="Output directory (default: results/extracted_results)")
    parser.add_argument("--models", "-m", nargs="+", 
                       default=["llama3.2:3b", "llama3.2:1b", "mistral:7b"],
                       help="Models to use for extraction")
    parser.add_argument("--limit", "-l", type=int, default=None,
                       help="Limit number of notes to process (for testing)")
    parser.add_argument("--rows", "-r", nargs="+", type=int, default=None,
                       help="Specific row indices to process (0-based, e.g., --rows 0 5 10)")
    parser.add_argument("--row-range", nargs=2, type=int, metavar=("START", "END"), default=None,
                       help="Process rows in range [START, END) (0-based, e.g., --row-range 10 20)")
    parser.add_argument("--mrns", nargs="+", default=None,
                       help="Specific MRNs to process (e.g., --mrns 907009329 907163230)")
    parser.add_argument("--random-sample", type=int, default=None,
                       help="Process random sample of N rows")
    parser.add_argument("--analyze-results", action="store_true",
                       help="Analyze existing extraction results")
    
    args = parser.parse_args()
    
    try:
        if args.analyze_results:
            # Analyze existing results
            analyze_extraction_results(args.output)
        else:
            # Load primary notes data
            notes_df = load_notes_data(args.input)
            
            # Load secondary DC summary data if provided
            dc_summary_df = None
            if args.dc_summary:
                try:
                    dc_summary_df = load_dc_summary_data(args.dc_summary)
                    print(f"Two-step extraction enabled: H&P notes + DC summary")
                except Exception as e:
                    print(f"Warning: Failed to load DC summary data: {e}")
                    print("Proceeding with H&P notes only")
            else:
                print("Single-step extraction: H&P notes only")
            
            # Extract baseline information
            extract_baseline_info(
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
        print(f"Extraction failed: {e}")
        raise


def analyze_extraction_results(output_dir: str = "results/extracted_results") -> None:
    """Analyze extraction results to understand success/failure patterns"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Output directory {output_dir} not found")
        return
    
    result_files = list(output_path.glob("baseline_extraction_*.csv"))
    
    if not result_files:
        print("No extraction result files found")
        return
    
    print("\n" + "="*80)
    print("EXTRACTION RESULTS ANALYSIS")
    print("="*80)
    
    for file in result_files:
        model_name = file.stem.replace("baseline_extraction_", "")
        df = pd.read_csv(file)
        
        print(f"\n{model_name}:")
        print(f"  Total records: {len(df)}")
        
        # Check extraction success
        if 'extraction_success' in df.columns:
            successful = df['extraction_success'].sum()
            print(f"  Successful extractions: {successful}/{len(df)} ({successful/len(df)*100:.1f}%)")
        
        # Count empty rows (only MRN + metadata)
        baseline_cols = [col for col in BASELINE_FIELDS if col in df.columns and col != 'MRN']
        empty_rows = df[baseline_cols].isna().all(axis=1).sum()
        print(f"  Completely empty rows: {empty_rows}")
        
        # Count rows with some data
        partial_rows = 0
        if baseline_cols:
            for _, row in df.iterrows():
                non_null_count = sum(1 for col in baseline_cols if pd.notna(row[col]) and row[col] != '')
                if 0 < non_null_count < len(baseline_cols):
                    partial_rows += 1
        
        print(f"  Partially filled rows: {partial_rows}")
        print(f"  Fully populated rows: {len(df) - empty_rows - partial_rows}")
        
        # Show field-by-field extraction rates
        if baseline_cols:
            print("  Field extraction rates:")
            for field in baseline_cols:
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
