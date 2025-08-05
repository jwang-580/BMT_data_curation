import json
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field, validator
from ollama import chat
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OAI_API_KEY'))

# Available local LLM models configuration
AVAILABLE_MODELS = {
    'llama3.3': 'llama3.3',
    'llama3.3-q8': 'llama3.3:70b-instruct-q8_0',
    'gemma3_fp16': 'gemma3:27b-it-fp16',
    'gemma3_q8': 'gemma3:27b-it-q8_0',
    'deepseek-r1': 'deepseek-r1:70b'
}

# Pydantic models for structured output
class HomeMeds(BaseModel):
    medications: List[str] = Field(description="List of medications with dosage forms, e.g. 'Celexa 20mg capsule'")

class AdmitMeds(BaseModel):
    medications: List[str] = Field(description="List of new medication names with route of administration, e.g. 'Propofol, IV'")

class ScanEntry(BaseModel):
    date: datetime = Field(description="date of the scan in format '%Y-%m-%d %H:%M:%S', if not mentioned, use the admission datetime")
    scan_results: str = Field(description="results of the scan")

class Scans(BaseModel):
    scans: List[ScanEntry] = Field(description="List of scans with their dates and results")

class ModifiedEvent(BaseModel):
    modified_text: str = Field(description="Event text with medication mentions replaced")

def process_admission_info(hadm_id: float, model_name: str = 'gemma3_q8') -> Tuple[List[Dict], List[Dict]]:
    """
    Process admission medications for a single HADM_ID.
    
    Args:
        hadm_id (float): The hospital admission ID to process
        model_name (str): The name of the LLM model to use (must be in AVAILABLE_MODELS)
    
    Returns:
        Tuple[List[Dict], List[Dict]]: Admission dataset rows and progress dataset rows
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    model = AVAILABLE_MODELS[model_name]
    
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
    
    # Load and process hospital medications
    hospital_meds_med_list = json.load(open(f'results/notes/sample_meds_{hadm_id}.json'))
    
    # Load events to get admission time
    events = json.load(open(f'results/notes/events_{hadm_id}.json'))
    admit_time = datetime.strptime(events[0]['admit_time'], '%Y-%m-%d %H:%M:%S')
    discharge_time = datetime.strptime(events[0]['discharge_time'], '%Y-%m-%d')
    
    # Process admission day medications
    admission_day_meds = []
    for med in hospital_meds_med_list:
        if 'start_time' in med and med['start_time']:
            med_start_time = datetime.strptime(med['start_time'], '%Y-%m-%d %H:%M:%S')
            if med_start_time < admit_time + timedelta(hours=6):
                admission_day_meds.append(med)
    
    # Load and process scans
    scan_notes = json.load(open(f'results/notes/scans_{hadm_id}.json'))
    scan_prompt = f"""
    Extract all medical imaging scan results as well as the DATE from the below json file. 
    If the date of the scan is not mentioned, use the admission datetime {admit_time}.
    Only inlude radiology imaging results. Do not include any other tests.
    Scans:
    {scan_notes}
    """
    
    response = chat(
        messages=[{'role': 'user', 'content': scan_prompt}],
        model=model,
        format=Scans.model_json_schema(),
    )
    scan_dict = Scans.model_validate_json(response.message.content)
    
    # Process admission scans
    admission_scans = {}
    for scan in scan_dict.scans:
        try:
            scan_date = scan.date.replace(tzinfo=None)  # Convert to naive datetime
            if admit_time - timedelta(days=1) < scan_date < admit_time + timedelta(hours=6):
                admission_scans[scan_date.strftime('%Y-%m-%d %H:%M:%S')] = scan.scan_results
        except Exception as e:
            pass
    
    # Process labs
    lab_dict = json.load(open(f'results/notes/sample_lab_{hadm_id}.json'))
    admission_labs = []
    baseline_labs = []
    
    for lab in lab_dict:
        lab_time = datetime.strptime(lab['lab_time'], '%Y-%m-%d %H:%M:%S')
        lab_entry = {
            'lab': lab['lab_name'] + ', ' + lab['lab_value'],
            'lab_time': lab['lab_time']
        }
        
        if admit_time - timedelta(hours=24) < lab_time < admit_time + timedelta(hours=6):
            admission_labs.append(lab_entry)
        
        if lab_time < admit_time - timedelta(days=2):
            baseline_labs.append(lab_entry)
    
    # Sort and process baseline labs
    baseline_labs.sort(key=lambda x: datetime.strptime(x['lab_time'], '%Y-%m-%d %H:%M:%S'), reverse=True)
    baseline_labs = baseline_labs[:10]
    
    # Remove lab_time from lab entries
    for lab in admission_labs + baseline_labs:
        lab.pop('lab_time', None)
    
    # Process new medications
    admission_day_meds_list = [med['medication'] for med in admission_day_meds]
    new_meds_prompt = f"""
    You are provided with two medication lists. 
    Identify and return NEW and DIFFERENT medications in LIST 2 compared to LIST 1. 
    Return a list of medication names and route of administration, EXACTLY as they appear in LIST 2, e.g. "Propofol, IV".
    
    LIST 1: 
    {home_meds_med_list}
    
    LIST 2:
    {admission_day_meds_list}
    """
    
    response = chat(
        messages=[{'role': 'user', 'content': new_meds_prompt}],
        model=model,
        format=AdmitMeds.model_json_schema(),
    )
    new_admission_meds = AdmitMeds.model_validate_json(response.message.content)
    
    # Filter medications
    filtered_meds_prompt = f"""
    Review and REMOVE medications that are:
    1. IV fluids (such as NS, D5W)
    2. Used for intubation (e.g. midazolam, vecuronium, propofol)
    3. vasopressors
    4. Used for pain (e.g. fentanyl, opioids)
    5. Supportive meds for symptom control (e.g. senna, vitamins, tylenol, electrolyte replacement, DVT prophylaxis, antiemetics)
    In other words, only retain medications that are essential for the patient's treatment.

    Return a list of medication names and route of administration, EXACTLY as they appear under key "medication" in the LIST, e.g. "Propofol, IV"
    
    LIST:
    {[med['medication'] for med in hospital_meds_med_list]}
    """
    
    response = chat(
        messages=[{'role': 'user', 'content': filtered_meds_prompt}],
        model=model,
        format=AdmitMeds.model_json_schema(),
    )
    filtered_all_hospital_meds = AdmitMeds.model_validate_json(response.message.content)
    
    # Instead of calling create_admission_dataset, prepare the data
    admission_rows = create_admission_dataset(hadm_id, admit_time, admission_day_meds, filtered_all_hospital_meds,
                                           home_meds_med_list, baseline_labs, admission_labs, admission_scans)
    
    # Instead of calling create_progress_dataset, prepare the data
    progress_rows = create_progress_dataset(hadm_id, admit_time, discharge_time, events, lab_dict,
                                         scan_dict, hospital_meds_med_list, filtered_all_hospital_meds, model)
    
    return admission_rows, progress_rows

def create_admission_dataset(hadm_id, admit_time, admission_day_meds, filtered_all_hospital_meds,
                           home_meds_med_list, baseline_labs, admission_labs, admission_scans) -> List[Dict]:
    """
    Create dataset for admission medications analysis
    
    Returns:
        List[Dict]: List of rows for the admission dataset
    """
    # Load HPI
    HPI = json.load(open(f'results/notes/HPI_{hadm_id}.json'))
    HPI_text = HPI[0]['0']
    chief_complaint_match = re.search(r'(Chief Complaint:[\s\S]*)', HPI_text, re.DOTALL)
    HPI = chief_complaint_match.group(1).strip() if chief_complaint_match else HPI_text
    
    admission_info = f"""
    Below is admission information for a patient.
    
    HPI:
    {HPI}
    
    Home meds:
    {home_meds_med_list}
    
    Abnormal labs at baseline:
    {[lab['lab'] for lab in baseline_labs]}
    
    Abnormal labs at admission:
    {[lab['lab'] for lab in admission_labs]}
    
    Scans done at admission:
    {admission_scans}
    
    What medication should be started for this patient at admission?
    """
    
    rows = []
    filtered_meds = [med['medication'] for med in admission_day_meds 
                    if med['medication'].lower() in [m.lower() for m in filtered_all_hospital_meds.medications]]
    
    for med_name in set(filtered_meds):
        rows.append({
            'HADM_ID': hadm_id,
            'admit_time': admit_time,
            'HPI': admission_info,
            'medication': med_name,
        })
    
    return rows

def create_progress_dataset(hadm_id, admit_time, discharge_time, events, lab_dict,
                          scan_dict, hospital_meds_med_list, filtered_all_hospital_meds, model=None) -> List[Dict]:
    """
    Create dataset for hospital progress medications analysis
    
    Returns:
        List[Dict]: List of rows for the progress dataset
    """
    current_time = admit_time + timedelta(hours=6)
    rows = []
    
    # Process events to remove medication mentions
    processed_events = []
    for event in events:
        if 'events' not in event or not event['events']:
            processed_events.append(event)
            continue
            
        event_text = event.get('events', '')
        if model:
            # Process event text to replace medication mentions
            single_event_prompt = f"""
            Modify the following hospital event text by replacing any mentions of specific medications with the generic word 'medication'.
            Do not remove or alter any other information such as dates, vitals, or non-medication related events.
            
            Event Text:
            {event_text}
            """
            
            try:
                response = chat(
                    messages=[{'role': 'user', 'content': single_event_prompt}],
                    model=model,
                    format=ModifiedEvent.model_json_schema(),
                )
                
                result = ModifiedEvent.model_validate_json(response.message.content)
                
                # Create a copy of the original event and update only the events field
                modified_event = event.copy()
                modified_event['events'] = result.modified_text
                processed_events.append(modified_event)
            except Exception as e:
                print(f"Error processing event: {str(e)}")
                processed_events.append(event)  # Use the original event if processing fails
        else:
            processed_events.append(event)  # Use the original event if no model provided
    
    while current_time <= discharge_time:
        next_time = current_time + timedelta(hours=24)
        
        # Process daily events using processed events
        daily_events = [event for event in processed_events 
                       if datetime.strptime(event['event_time'], '%Y-%m-%d %H:%M:%S') < next_time]
        
        # Process labs
        old_labs = []
        new_labs = []
        for lab in lab_dict:
            lab_time = datetime.strptime(lab['lab_time'], '%Y-%m-%d %H:%M:%S')
            lab_entry = {
                'lab': lab['lab_name'] + ', ' + lab['lab_value'],
                'lab_time': lab['lab_time']
            }
            if current_time - timedelta(hours=24) <= lab_time < current_time:
                old_labs.append(lab_entry)
            elif current_time <= lab_time < next_time:
                new_labs.append(lab_entry)
        
        # Remove lab_time
        for lab in old_labs + new_labs:
            lab.pop('lab_time', None)
        
        # Process medications
        current_meds = [med['medication'] for med in hospital_meds_med_list 
                       if (datetime.strptime(med['start_time'], '%Y-%m-%d %H:%M:%S') < current_time and
                           datetime.strptime(med['end_time'], '%Y-%m-%d %H:%M:%S') > current_time and
                           med['medication'].lower() in [m.lower() for m in filtered_all_hospital_meds.medications])]
        
        # Process scans in progress_scans
        progress_scans = {}
        for scan in scan_dict.scans:
            try:
                scan_date = scan.date.replace(tzinfo=None)  # Convert to naive datetime
                if scan_date < current_time:
                    progress_scans[scan_date.strftime('%Y-%m-%d %H:%M:%S')] = scan.scan_results
            except Exception as e:
                pass
        
        # Create progress note
        HPI = json.load(open(f'results/notes/HPI_{hadm_id}.json'))
        HPI_text = HPI[0]['0']
        chief_complaint_match = re.search(r'(Chief Complaint:[\s\S]*)', HPI_text, re.DOTALL)
        HPI = chief_complaint_match.group(1).strip() if chief_complaint_match else HPI_text

        progression_prompt = f"""
        Below is hospital daily progress for a patient.
        
        HPI:
        {HPI}
        
        Events since admission:
        {[{k: v for k, v in event.items() if k not in ['admit_time', 'vitals', 'discharge_time']} 
          for event in daily_events]}
        
        Recent vitals:
        {[event.get('vitals', '') for event in daily_events]}
        
        Yesterday's abnormal labs:
        {[lab['lab'] for lab in old_labs]}
        
        Today's abnormal labs:
        {[lab['lab'] for lab in new_labs]}
        
        Image studies since admission:
        {progress_scans}
        
        Current medications:
        {current_meds}
        
        What medication should be started for this patient today?
        """
        
        # Get new medications started on this day
        new_meds = set(med['medication'] for med in hospital_meds_med_list 
                      if (current_time < datetime.strptime(med['start_time'], '%Y-%m-%d %H:%M:%S') < next_time and
                          med['medication'].lower() in [m.lower() for m in filtered_all_hospital_meds.medications]))
        new_meds = [med for med in new_meds if med not in current_meds]
        
        for med_name in new_meds:
            rows.append({
                'HADM_ID': hadm_id,
                'current_time': current_time,
                'Progress_note': progression_prompt,
                'medication': med_name,
            })
        
        current_time = next_time
    
    return rows

def load_hadm_ids_from_json(json_path: str) -> List[float]:
    """
    Load HADM_IDs from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing HADM_IDs
        
    Returns:
        List[float]: List of HADM_IDs
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('hadm_ids', [])
    except Exception as e:
        print(f"Error loading HADM_IDs from {json_path}: {str(e)}")
        return []

def process_admission_info_gpt4(hadm_id: float, model_name: str = 'gemma3_q8') -> Tuple[List[Dict], List[Dict]]:
    """
    Process medications for a single HADM_ID using GPT-4.
    Process scans using a local LLM model.
    
    Args:
        hadm_id (float): The hospital admission ID to process
        model_name (str): The name of the local LLM model to use for scan processing
    
    Returns:
        Tuple[List[Dict], List[Dict]]: Admission dataset rows and progress dataset rows
    """
    # Validate the model_name
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    model = AVAILABLE_MODELS[model_name]
    
    # Load home medications
    home_meds = json.load(open(f'results/notes/admission_meds_{hadm_id}.json'))
    meds_prompt = f"""
    Extract all medications from the below json file, including the dosage form and dosage strength. e.g. 'Celexa 20mg capsule'
    Medications:
    {home_meds}
    """
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": meds_prompt}],
        response_format=HomeMeds
    )
    home_meds_med_list = response.choices[0].message.parsed
    
    # Load and process hospital medications
    hospital_meds_med_list = json.load(open(f'results/notes/sample_meds_{hadm_id}.json'))
    
    # Load events to get admission time
    events = json.load(open(f'results/notes/events_{hadm_id}.json'))
    admit_time = datetime.strptime(events[0]['admit_time'], '%Y-%m-%d %H:%M:%S')
    discharge_time = datetime.strptime(events[0]['discharge_time'], '%Y-%m-%d')
    
    # Process admission day medications
    admission_day_meds = []
    for med in hospital_meds_med_list:
        if 'start_time' in med and med['start_time']:
            med_start_time = datetime.strptime(med['start_time'], '%Y-%m-%d %H:%M:%S')
            if med_start_time < admit_time + timedelta(hours=6):
                admission_day_meds.append(med)
    
    # Load and process scans using LOCAL LLM
    scan_notes = json.load(open(f'results/notes/scans_{hadm_id}.json'))
    scan_prompt = f"""
    Extract all medical imaging scan results as well as the DATE from the below json file. 
    If the date of the scan is not mentioned, use the admission datetime {admit_time}.
    Only inlude radiology imaging results. Do not include any other tests.
    Scans:
    {scan_notes}
    """
    
    response = chat(
        messages=[{'role': 'user', 'content': scan_prompt}],
        model=model,
        format=Scans.model_json_schema(),
    )
    scan_dict = Scans.model_validate_json(response.message.content)
    
    # Process admission scans
    admission_scans = {}
    for scan in scan_dict.scans:
        try:
            scan_date = scan.date.replace(tzinfo=None)  # Convert to naive datetime
            if admit_time - timedelta(days=1) < scan_date < admit_time + timedelta(hours=6):
                admission_scans[scan_date.strftime('%Y-%m-%d %H:%M:%S')] = scan.scan_results
        except Exception as e:
            pass
    
    # Process labs
    lab_dict = json.load(open(f'results/notes/sample_lab_{hadm_id}.json'))
    admission_labs = []
    baseline_labs = []
    
    for lab in lab_dict:
        lab_time = datetime.strptime(lab['lab_time'], '%Y-%m-%d %H:%M:%S')
        lab_entry = {
            'lab': lab['lab_name'] + ', ' + lab['lab_value'],
            'lab_time': lab['lab_time']
        }
        
        if admit_time - timedelta(hours=24) < lab_time < admit_time + timedelta(hours=6):
            admission_labs.append(lab_entry)
        
        if lab_time < admit_time - timedelta(days=2):
            baseline_labs.append(lab_entry)
    
    # Sort and process baseline labs
    baseline_labs.sort(key=lambda x: datetime.strptime(x['lab_time'], '%Y-%m-%d %H:%M:%S'), reverse=True)
    baseline_labs = baseline_labs[:10]
    
    # Remove lab_time from lab entries
    for lab in admission_labs + baseline_labs:
        lab.pop('lab_time', None)
    
    # Process new medications with GPT-4
    admission_day_meds_list = [med['medication'] for med in admission_day_meds]
    new_meds_prompt = f"""
    You are provided with two medication lists. 
    Identify and return NEW and DIFFERENT medications in LIST 2 compared to LIST 1. 
    Return a list of medication names and route of administration, EXACTLY as they appear in LIST 2, e.g. "Propofol, IV".
    
    LIST 1: 
    {home_meds_med_list}
    
    LIST 2:
    {admission_day_meds_list}
    """
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": new_meds_prompt}],
        response_format=AdmitMeds
    )
    new_admission_meds = response.choices[0].message.parsed
    
    # Filter medications with GPT-4
    filtered_meds_prompt = f"""
    Review and REMOVE medications that are:
    1. IV fluids (such as NS, D5W)
    2. Used for intubation (e.g. midazolam, vecuronium, propofol)
    3. vasopressors
    4. Used for pain (e.g. fentanyl, opioids)
    5. Supportive meds for symptom control (e.g. senna, vitamins, tylenol, electrolyte replacement, DVT prophylaxis, antiemetics)
    In other words, only retain medications that are essential for the patient's treatment.

    Return a list of medication names and route of administration, EXACTLY as they appear under key "medication" in the LIST, e.g. "Propofol, IV"
    
    LIST:
    {[med['medication'] for med in hospital_meds_med_list]}
    """
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": filtered_meds_prompt}],
        response_format=AdmitMeds
    )
    filtered_all_hospital_meds = response.choices[0].message.parsed
    print(filtered_all_hospital_meds)
    
    # Instead of calling create_admission_dataset, prepare the data
    admission_rows = create_admission_dataset(hadm_id, admit_time, admission_day_meds, filtered_all_hospital_meds,
                                           home_meds_med_list, baseline_labs, admission_labs, admission_scans)
    
    # Instead of calling create_progress_dataset, prepare the data
    progress_rows = create_progress_dataset(hadm_id, admit_time, discharge_time, events, lab_dict,
                                         scan_dict, hospital_meds_med_list, filtered_all_hospital_meds, model)
    
    return admission_rows, progress_rows

def main(hadm_ids: List[float] = None, json_path: str = None, model_name: str = 'gemma3_q8', use_gpt4: bool = False):
    """
    Process admission medications for multiple HADM_IDs.
    
    Args:
        hadm_ids (List[float], optional): List of hospital admission IDs to process
        json_path (str, optional): Path to JSON file containing HADM_IDs
        model_name (str): Name of the LLM model to use
        use_gpt4 (bool): Whether to use GPT-4 instead of local LLM
    """
    # If hadm_ids is provided directly, use that and ignore json_path
    if hadm_ids:
        print(f"Using provided HADM_IDs from command line arguments...")
    # Otherwise try to load from JSON file
    elif json_path:
        print(f"Loading HADM_IDs from JSON file: {json_path}")
        hadm_ids = load_hadm_ids_from_json(json_path)
    
    if not hadm_ids:
        raise ValueError("No HADM_IDs provided. Please provide either --hadm_ids or --json_file")
    
    all_admission_rows = []
    all_progress_rows = []
    
    print(f"Processing {len(hadm_ids)} HADM_IDs using {'GPT-4 for medication analysis and local model ' + model_name + ' for scan processing' if use_gpt4 else f'local model {model_name}'}...")
    
    for hadm_id in hadm_ids:
        print(f"Processing HADM_ID: {hadm_id}")
        try:
            if use_gpt4:
                admission_rows, progress_rows = process_admission_info_gpt4(hadm_id, model_name)
            else:
                admission_rows, progress_rows = process_admission_info(hadm_id, model_name)
            all_admission_rows.extend(admission_rows)
            all_progress_rows.extend(progress_rows)
            print(f"Successfully processed HADM_ID: {hadm_id}")
        except Exception as e:
            print(f"Error processing HADM_ID {hadm_id}: {str(e)}")
    
    # Create output directory if it doesn't exist
    os.makedirs('results/datasets', exist_ok=True)
    
    # Create combined datasets
    if all_admission_rows:
        admission_df = pd.DataFrame(all_admission_rows)
        admission_df.to_csv('results/datasets/combined_admission_medications_analysis.csv', index=False)
        print(f"Created combined admission medications analysis file with {len(all_admission_rows)} entries")
    
    if all_progress_rows:
        progress_df = pd.DataFrame(all_progress_rows)
        progress_df.to_csv('results/datasets/combined_hospital_progress_medications_analysis.csv', index=False)
        print(f"Created combined hospital progress medications analysis file with {len(all_progress_rows)} entries")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process admission medications for multiple hospital admissions')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--hadm_ids', type=float, nargs='+',
                      help='List of hospital admission IDs to process')
    group.add_argument('--json_file', type=str, default='results/notes/selected_hadm_ids.json',
                      help='Path to JSON file containing HADM_IDs')
    parser.add_argument('--model', type=str, default='gemma3_q8',
                      choices=list(AVAILABLE_MODELS.keys()),
                      help='Local LLM model to use for processing')
    parser.add_argument('--use_gpt4', action='store_true',
                      help='Use GPT-4 for medication analysis while still using the local model for scan processing')
    
    args = parser.parse_args()
    
    # If no arguments provided, use default JSON file
    if not args.hadm_ids and not args.json_file:
        args.json_file = 'results/notes/selected_hadm_ids.json'
        print("No HADM_IDs provided, using default JSON file path:", args.json_file)
    
    main(hadm_ids=args.hadm_ids, json_path=args.json_file, model_name=args.model, use_gpt4=args.use_gpt4) 