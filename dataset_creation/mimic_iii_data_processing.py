import pandas as pd
import json
import re
import os
import random
from typing import List, Optional

def create_output_directory(base_dir: str = "results/notes") -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(base_dir, exist_ok=True)

def load_filtered_data(data_dir: str) -> tuple:
    """Load pre-filtered datasets if they exist."""
    try:
        filtered_notes = pd.read_csv(os.path.join(data_dir, 'filtered_notes.csv'))
        labs = pd.read_csv(os.path.join(data_dir, 'labs_with_labels.csv'))
        prescription = pd.read_csv(os.path.join(data_dir, 'PRESCRIPTIONS.csv'))
        available_hadm_ids = list(set(filtered_notes['HADM_ID']))
        return filtered_notes, prescription, labs, available_hadm_ids
    except FileNotFoundError as e:
        print(f"Could not find pre-filtered data: {e}")
        return None

def load_raw_data() -> tuple:
    """Load and filter the initial datasets."""
    note = pd.read_csv('data/NOTEEVENTS.csv')
    
    note_type = ['Discharge summary', 'Physician ', 'Consult', 'Radiology']
    note = note[note['CATEGORY'].isin(note_type)]
    
    # First find all admission and discharge notes
    admission_notes = note[note['DESCRIPTION'].str.contains('Resident Admission Note', na=False)]
    admission_notes = admission_notes[~admission_notes['DESCRIPTION'].str.contains('Surgical Admission Note', na=False)]
    discharge_notes = note[note['CATEGORY'] == 'Discharge summary']
    
    # Get HADM_IDs that have both types of notes
    hadm_ids_with_admission = set(admission_notes['HADM_ID'])
    hadm_ids_with_discharge = set(discharge_notes['HADM_ID'])
    admissions_with_both = hadm_ids_with_admission & hadm_ids_with_discharge
    
    # Create a filtered dataset that includes the admission and discharge notes
    filtered_notes = pd.concat([
        admission_notes[admission_notes['HADM_ID'].isin(admissions_with_both)],
        discharge_notes[discharge_notes['HADM_ID'].isin(admissions_with_both)],
        note[(note['CATEGORY'].isin(['Physician ', 'Consult', 'Radiology'])) & 
             (note['HADM_ID'].isin(admissions_with_both))]
    ]).drop_duplicates()
    
    prescription = pd.read_csv('data/PRESCRIPTIONS.csv')
    
    labevents = pd.read_csv('data/LABEVENTS.csv')
    lab_labels = pd.read_csv('data/D_LABITEMS.csv')
    lab_labels['label'] = lab_labels['LABEL'] + ', ' + lab_labels['FLUID']
    lab_labels = lab_labels[['ITEMID', 'label']]
    labs = labevents.merge(lab_labels, on='ITEMID', how='left')
    
    # Verify each HADM_ID in the list actually has both types of notes in filtered_notes
    final_hadm_ids = []
    for hadm_id in admissions_with_both:
        hadm_notes = filtered_notes[filtered_notes['HADM_ID'] == hadm_id]
        has_admission = any(hadm_notes['DESCRIPTION'].str.contains('Resident Admission Note', na=False))
        has_discharge = any(hadm_notes['CATEGORY'] == 'Discharge summary')
        if has_admission and has_discharge:
            final_hadm_ids.append(hadm_id)
    
    print(f"Found {len(final_hadm_ids)} HADM_IDs with both admission and discharge notes")
    return filtered_notes, prescription, labs, final_hadm_ids

def save_filtered_data(filtered_notes: pd.DataFrame, labs: pd.DataFrame, prescription: pd.DataFrame, data_dir: str) -> None:
    """Save filtered datasets for future use."""
    os.makedirs(data_dir, exist_ok=True)
    filtered_notes.to_csv(os.path.join(data_dir, 'filtered_notes.csv'), index=False)
    labs.to_csv(os.path.join(data_dir, 'labs_with_labels.csv'), index=False)
    prescription.to_csv(os.path.join(data_dir, 'PRESCRIPTIONS.csv'), index=False)
    print(f"Saved filtered data to {data_dir}")

def process_admission(adm_id: int, filtered_notes: pd.DataFrame, prescription: pd.DataFrame, 
                     labs: pd.DataFrame, output_dir: str) -> None:
    """Process data for a single admission ID."""
    # Filter progress notes
    progress_notes = filtered_notes[filtered_notes['HADM_ID'] == adm_id]
    progress_notes = progress_notes.drop_duplicates(subset=['CGID'])
    progress_notes = progress_notes.drop_duplicates(subset=['CHARTTIME'])
    
    # Get admission and discharge times
    admit_note = progress_notes[progress_notes['DESCRIPTION'].str.contains('Resident Admission Note', na=False)]
    if admit_note.empty:
        print(f"No admission note found for HADM_ID {adm_id}")
        return
        
    admit_time = admit_note['CHARTTIME'].iloc[0]
    discharge_time = progress_notes[progress_notes['CATEGORY'].str.contains('Discharge summary', na=False)]['CHARTDATE'].item()
    pt_id = admit_note['SUBJECT_ID'].iloc[0]
    
    # Process medications
    sample_meds = prescription[prescription['HADM_ID']==adm_id]
        
    # Process labs
    sample_lab = labs[(labs['SUBJECT_ID'] == pt_id) & (labs['FLAG'] == 'abnormal')]
    
    # Process discharge summary
    discharge_summary = progress_notes[progress_notes['CATEGORY'] == 'Discharge summary']
    if not discharge_summary.empty:
        # Extract HPI
        HPI = discharge_summary['TEXT'].str.extract(r'Admission Date:([\s\S]*)Social History:')
        if not HPI.empty:
            HPI[0] = HPI[0].str.replace('\n', ' ')
            HPI[0] = HPI[0].str.replace('\/', '')
            HPI.to_json(f'{output_dir}/HPI_{adm_id}.json', orient='records')
        
        # Extract admission medications
        admission_meds = discharge_summary['TEXT'].str.extract(r'Medications on Admission:([\s\S]*)Discharge Medications:')
        if not admission_meds.empty:
            admission_meds = admission_meds[0].str.replace('\n', ',')
            admission_meds.to_json(f'{output_dir}/admission_meds_{adm_id}.json', orient='records')
        
        # Extract scan results
        scans = discharge_summary['TEXT'].str.extract(r'Pertinent Results:([\s\S]*)Brief Hospital Course:')
        if not scans.empty:
            scans[0] = scans[0].str.replace('\n', ' ')
            scans[0] = scans[0].str.replace('\/', '')
            scans.to_json(f'{output_dir}/scans_{adm_id}.json', orient='records')
        
        # Extract assessment/plan
        a_p = discharge_summary['TEXT'].str.extract(r'Brief Hospital Course:([\s\S]*)Medications on Admission:')
        if not a_p.empty:
            a_p[0] = a_p[0].str.replace('\n', ' ')
            a_p.to_json(f'{output_dir}/a_p_{adm_id}.json', orient='records')
    
    # Process medications with timestamps
    meds_list = []
    for _, row in sample_meds.iterrows():
        # Handle null values and type conversions for medication fields
        dose_val = str(row['DOSE_VAL_RX']) if pd.notna(row['DOSE_VAL_RX']) else ''
        dose_unit = str(row['DOSE_UNIT_RX']) if pd.notna(row['DOSE_UNIT_RX']) else ''
        form_val = str(row['FORM_VAL_DISP']) if pd.notna(row['FORM_VAL_DISP']) else ''
        form_unit = str(row['FORM_UNIT_DISP']) if pd.notna(row['FORM_UNIT_DISP']) else ''
        
        # Construct dosage string only with available components
        dosage_parts = []
        if dose_val and dose_unit:
            dosage_parts.append(f"{dose_val} {dose_unit}")
        if form_val and form_unit:
            dosage_parts.append(f"{form_val} {form_unit}")
        dosage = ', '.join(dosage_parts) if dosage_parts else 'Not specified'
        
        med_dict = {
            'medication': f"{str(row['DRUG']) if pd.notna(row['DRUG']) else 'Unknown'}, {str(row['ROUTE']) if pd.notna(row['ROUTE']) else 'Unknown route'}",
            'start_time': row['STARTDATE'],
            'end_time': row['ENDDATE'],
            'dosage': dosage
        }
        meds_list.append(med_dict)
    
    with open(f'{output_dir}/sample_meds_{adm_id}.json', 'w') as f:
        json.dump(meds_list, f, indent=4)
    
    # Process labs with timestamps
    labs_list = []
    for _, row in sample_lab.iterrows():
        lab_dict = {
            'lab_name': row['label'],
            'lab_value': f"{str(row['VALUENUM'])} {str(row['VALUEUOM']).strip()}" if pd.notna(row['VALUEUOM']) else str(row['VALUENUM']),
            'lab_time': row['CHARTTIME']
        }
        labs_list.append(lab_dict)
    
    with open(f'{output_dir}/sample_lab_{adm_id}.json', 'w') as f:
        json.dump(labs_list, f, indent=4)
    
    # Process events and vitals
    events_list = []
    daily_progress = progress_notes[progress_notes['DESCRIPTION'].str.contains('Progress Note', na=False)]
    
    for _, row in daily_progress.iterrows():
        text = row['TEXT'].replace('\n', ',')
        event_dict = {
            'admit_time': admit_time,
            'discharge_time': discharge_time,
            'event_time': row['CHARTTIME'],
            'vitals': re.search(r'(Tmax: [\s\S]*)RR:', text).group(1) if re.search(r'(Tmax: [\s\S]*)RR:', text) else None,
            'events': re.search(r'(24 Hour Events:[\s\S]*)Allergies:', text).group(1) if re.search(r'(24 Hour Events:[\s\S]*)Allergies:', text) else None
        }
        events_list.append(event_dict)
    
    with open(f'{output_dir}/events_{adm_id}.json', 'w') as f:
        json.dump(events_list, f, indent=4)

def main(num_admissions: Optional[int] = None, output_dir: str = "results/notes", 
         use_filtered_data: bool = False, filtered_data_dir: str = "data",
         random_seed: Optional[int] = None):
    """Main function to process multiple admission IDs."""
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Load data
    if use_filtered_data:
        print("Attempting to load pre-filtered data...")
        result = load_filtered_data(filtered_data_dir)
        if result is not None:
            filtered_notes, prescription, labs, available_hadm_ids = result
            print("Successfully loaded pre-filtered data")
        else:
            print("Falling back to loading raw data...")
            filtered_notes, prescription, labs, available_hadm_ids = load_raw_data()
            print("Saving filtered data for future use...")
            save_filtered_data(filtered_notes, labs, prescription, filtered_data_dir)
    else:
        print("Loading raw data...")
        filtered_notes, prescription, labs, available_hadm_ids = load_raw_data()
        print("Saving filtered data for future use...")
        save_filtered_data(filtered_notes, labs, prescription, filtered_data_dir)
    
    # Determine how many admissions to process
    total_available = len(available_hadm_ids)
    if num_admissions is None:
        num_admissions = total_available
    else:
        num_admissions = min(num_admissions, total_available)
    
    # Randomly select admission IDs
    selected_hadm_ids = random.sample(available_hadm_ids, num_admissions)
    
    # Save selected HADM_IDs for reproducibility
    with open(os.path.join(output_dir, 'selected_hadm_ids.json'), 'w') as f:
        json.dump({
            'random_seed': random_seed,
            'total_available': total_available,
            'num_selected': num_admissions,
            'hadm_ids': selected_hadm_ids
        }, f, indent=4)
    
    # Process selected admissions
    for i, adm_id in enumerate(selected_hadm_ids):
        print(f"Processing admission {i+1}/{num_admissions} (HADM_ID: {adm_id})")
        process_admission(adm_id, filtered_notes, prescription, labs, output_dir)
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process MIMIC-III notes data')
    parser.add_argument('--num_admissions', type=int, default=None,
                      help='Number of admission IDs to process (default: all available)')
    parser.add_argument('--output_dir', type=str, default="results/notes",
                      help='Output directory for processed files (default: results/notes)')
    parser.add_argument('--use_filtered_data', action='store_true',
                      help='Use pre-filtered data if available (default: False)')
    parser.add_argument('--filtered_data_dir', type=str, default="data",
                      help='Directory containing pre-filtered data (default: data)')
    parser.add_argument('--random_seed', type=int, default=None,
                      help='Random seed for reproducible selection (default: None)')
    
    args = parser.parse_args()
    main(args.num_admissions, args.output_dir, args.use_filtered_data, args.filtered_data_dir, args.random_seed) 