from ollama import chat
from pydantic import BaseModel, Field
from typing import List, Dict
import json
from datetime import datetime, timedelta
import pandas as pd
import re
import os

# Define model configurations
AVAILABLE_MODELS = {
    'llama3.3': 'llama3.3',
    'llama3.3-q8': 'llama3.3:70b-instruct-q8_0',
    'gemma3_fp16': 'gemma3:27b-it-fp16',
    'gemma3_q8': 'gemma3:27b-it-q8_0',
    'deepseek-r1': 'deepseek-r1:70b'
}

DEFAULT_MODEL = 'gemma3_q8'
model = AVAILABLE_MODELS[DEFAULT_MODEL]

# Define Pydantic models
class HomeMeds(BaseModel):
    medications: List[str] = Field(description="List of medications with dosage forms, e.g. 'Celexa 20mg capsule'")

class Scans(BaseModel):
    scans: Dict[str, str] = Field(description="Dictionary of scans with date and results")

class ModifiedEvent(BaseModel):
    modified_text: str = Field(description="Event text with medication mentions replaced")

class AdmitMeds(BaseModel):
    medications: List[str] = Field(description="List of new medication names, e.g. 'Propofol'")

def process_admission_meds(hadm_id: float) -> None:
    """Process admission medications for a single HADM_ID."""
    try:
        # Load home medications
        home_meds = json.load(open(f'results/notes/admission_meds_{hadm_id}.json'))
        
        meds_prompt = f"""
        Extract all medications from the below json file, including the dosage form and dosage strength. e.g. 'Celexa 20mg capsule'
        Medications:
        {home_meds}
        """
        
        response = chat(
            messages=[{'role': 'user', 'content': meds_prompt}],
            model=model,
            format=HomeMeds.model_json_schema(),
        )
        
        home_meds_med_list = HomeMeds.model_validate_json(response.message.content)
        
        # Load events to get admission time and discharge time
        events = json.load(open(f'results/notes/events_{hadm_id}.json'))
        admit_time = datetime.strptime(events[0]['admit_time'], '%Y-%m-%d %H:%M:%S')
        discharge_time = datetime.strptime(events[0]['discharge_time'], '%Y-%m-%d')
        
        # Process hospital medications for each day
        hospital_meds_med_list = json.load(open(f'results/notes/sample_meds_{hadm_id}.json'))
        
        # Iterate through each day of hospitalization
        current_time = admit_time
        while current_time <= discharge_time:
            next_time = current_time + timedelta(hours=24)
            
            # Get medications for this 24-hour period
            daily_meds = []
            for med in hospital_meds_med_list:
                if 'start_time' in med and med['start_time']:
                    med_start_time = datetime.strptime(med['start_time'], '%Y-%m-%d %H:%M:%S')
                    if current_time <= med_start_time < next_time:
                        daily_meds.append(med)
            
            # Load and process scans for this day
            scan_notes = json.load(open(f'results/notes/scans_{hadm_id}.json'))
            scan_prompt = f"""
            Extract all medical imaging scan results and date from the below json file.
            Scans:
            {scan_notes}
            """
            
            response = chat(
                messages=[{'role': 'user', 'content': scan_prompt}],
                model=model,
                format=Scans.model_json_schema(),
            )
            
            scan_dict = Scans.model_validate_json(response.message.content)
            
            # Process labs for this day
            lab_dict = json.load(open(f'results/notes/sample_lab_{hadm_id}.json'))
            daily_labs = []
            baseline_labs = []
            
            for lab in lab_dict:
                lab_time = datetime.strptime(lab['lab_time'], '%Y-%m-%d %H:%M:%S')
                lab_entry = {
                    'lab_name': lab['lab_name'],
                    'lab_value': lab['lab_value'],
                    'lab_time': lab['lab_time']
                }
                
                if current_time <= lab_time < next_time:
                    daily_labs.append(lab_entry)
                
                if lab_time < admit_time - timedelta(days=2):
                    baseline_labs.append(lab_entry)
            
            # Sort and limit baseline labs
            baseline_labs.sort(key=lambda x: datetime.strptime(x['lab_time'], '%Y-%m-%d %H:%M:%S'), reverse=True)
            baseline_labs = baseline_labs[:10]
            
            # Remove lab_time from lab entries
            for lab in daily_labs + baseline_labs:
                lab.pop('lab_time', None)
            
            # Process new medications for this day
            if daily_meds:
                new_meds_prompt = f"""
                You are provided with two medication lists. 
                Identify and return NEW and DIFFERENT medications in LIST 2 compared to LIST 1. 
                Return a list of medication names, EXACTLY as they are in LIST 2, e.g. "Propofol"

                LIST 1: 
                {home_meds_med_list}

                LIST 2:
                {daily_meds}
                """
                
                response = chat(
                    messages=[{'role': 'user', 'content': new_meds_prompt}],
                    model=model,
                    format=AdmitMeds.model_json_schema(),
                )
                
                parsed_meds = AdmitMeds.model_validate_json(response.message.content)
                
                # Filter medications
                filtered_meds_prompt = f"""
                You are provided with a list of medications.
                Review and remove medications that are 
                1. IV fluids (such as NS, D5W). 
                2. used for intubation (e.g. midazolam, vecuronium, propofol). 
                3. used for pain (e.g. fentanyl, opioids). 
                4. supportive meds (e.g. senna, vitamins, tylenol, electrolyte replacement, DVT prophylaxis)
                In other words, only include medications that are essential for the patient's treatment.

                LIST:
                {parsed_meds}
                """
                
                response_2 = chat(
                    messages=[{'role': 'user', 'content': filtered_meds_prompt}],
                    model=model,
                    format=AdmitMeds.model_json_schema(),
                )
                
                filtered_parsed_meds = AdmitMeds.model_validate_json(response_2.message.content)
                new_meds_med_list = [med.split(',')[0].lower() for med in filtered_parsed_meds.medications]
                
                # Match new meds to daily meds
                new_daily_meds = []
                for med in daily_meds:
                    if med['medication'].split(',')[0].lower() in new_meds_med_list:
                        new_daily_meds.append(med)
                
                # Load and process HPI
                HPI = json.load(open(f'results/notes/HPI_{hadm_id}.json'))
                HPI_text = HPI[0]['0']
                chief_complaint_match = re.search(r'(Chief Complaint:[\s\S]*)', HPI_text, re.DOTALL)
                HPI = chief_complaint_match.group(1).strip() if chief_complaint_match else HPI_text
                
                # Create daily information
                daily_info = f"""
                Below is information for hospital day {(current_time - admit_time).days + 1}.

                HPI:
                {HPI}

                Home meds:
                {home_meds_med_list}

                Labs from this 24-hour period:
                {daily_labs}

                Baseline labs:
                {baseline_labs}

                Scans from this 24-hour period:
                {daily_scans}

                What medication should be started for this patient on this day?
                """
                
                # Create output rows
                rows = []
                unique_meds = set(med['medication'] for med in new_daily_meds)
                for med_name in unique_meds:
                    row = {
                        'HADM_ID': hadm_id,
                        'hospital_day': (current_time - admit_time).days + 1,
                        'day_start_time': current_time,
                        'day_end_time': next_time,
                        'HPI': daily_info,
                        'medication': med_name,
                    }
                    rows.append(row)
                
                # Save to CSV
                if rows:
                    df = pd.DataFrame(rows)
                    os.makedirs('results/datasets', exist_ok=True)
                    output_file = f'results/datasets/medications_analysis_{hadm_id}_day_{(current_time - admit_time).days + 1}.csv'
                    df.to_csv(output_file, index=False)
                    print(f"Created CSV file: {output_file}")
            
            # Move to next day
            current_time = next_time
        
    except Exception as e:
        print(f"Error processing HADM_ID {hadm_id}: {str(e)}")

def process_multiple_admissions(hadm_ids: List[float]) -> None:
    """Process multiple HADM_IDs."""
    for hadm_id in hadm_ids:
        print(f"\nProcessing HADM_ID: {hadm_id}")
        process_admission_meds(hadm_id)

if __name__ == "__main__":
    # Example usage
    hadm_ids = [152136.0]  # Replace with your list of HADM_IDs
    process_multiple_admissions(hadm_ids) 