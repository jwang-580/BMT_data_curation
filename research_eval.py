import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import argparse

RESEARCH_FIELDS = ['crs_y_n', 'fever_onset_date', 'last_fever_date', 'max_temp', 'hypotension_y_n',
                   'pressor_use_num', 'hypoxia_y_n', 'high_flow_o2_y_n', 'bipap_or_intubation_y_n', 
                   'neurotox_y_n', 'toci_y_n', 'toci_start_date', 'toci_stop_date', 'total_dose_toci']

def compare_models_results(output_dir: str = "extracted_results") -> None:
    """Compare results from different models"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Output directory {output_dir} not found")
        return
    
    # Find all result files
    result_files = list(output_path.glob("research_extraction_*.csv"))
    
    if not result_files:
        print("No extraction result files found")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Load and compare results
    comparison_data = []
    
    for file in result_files:
        model_name = file.stem.replace("research_extraction_", "")
        df = pd.read_csv(file)
        
        # Calculate statistics
        stats = {
            'model': model_name,
            'total_records': len(df),
            'successful_extractions': df['extraction_success'].sum(),
            'success_rate': df['extraction_success'].mean(),
        }
        
        # Count non-null extractions for each field
        for field in RESEARCH_FIELDS:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                stats[f'{field}_extracted'] = non_null_count
                stats[f'{field}_rate'] = non_null_count / len(df)
        
        comparison_data.append(stats)
    
    # Save comparison report
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = output_path / "research_model_comparison_report.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"Model comparison report saved to: {comparison_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("RESEARCH MODEL COMPARISON SUMMARY")
    print("="*80)
    
    for _, row in comparison_df.iterrows():
        print(f"\nModel: {row['model']}")
        print(f"  Success Rate: {row['success_rate']:.2%} ({row['successful_extractions']}/{row['total_records']})")
        
        # Show top extracted fields
        field_rates = [(col, row[col]) for col in row.index if col.endswith('_rate') and col != 'success_rate']
        field_rates.sort(key=lambda x: x[1], reverse=True)
        
        print("  Top extracted fields:")
        for field, rate in field_rates[:5]:
            field_name = field.replace('_rate', '')
            print(f"    {field_name}: {rate:.2%}")


def load_ground_truth(ground_truth_path: str) -> pd.DataFrame:
    """Load ground truth data for evaluation"""
    try:
        df = pd.read_csv(ground_truth_path)
        print(f"Loaded ground truth data: {len(df)} records")
        return df
    except Exception as e:
        print(f"Failed to load ground truth data: {e}")
        raise


def calculate_field_metrics(predicted: pd.Series, ground_truth: pd.Series, field_name: str) -> Dict[str, float]:
    """Calculate precision, recall, F1 for a specific field"""
    
    # Define field types
    yn_fields = ['crs_y_n', 'hypotension_y_n', 'hypoxia_y_n', 'high_flow_o2_y_n', 'bipap_or_intubation_y_n', 'neurotox_y_n', 'toci_y_n']
    numeric_fields = ['max_temp', 'pressor_use_num', 'total_dose_toci']
    date_fields = ['fever_onset_date', 'last_fever_date', 'toci_start_date', 'toci_stop_date']
    
    if field_name in numeric_fields:
        # Handle numeric fields specially
        pred_numeric = pd.to_numeric(predicted, errors='coerce')
        truth_numeric = pd.to_numeric(ground_truth, errors='coerce')
        
        # Special handling for max_temp: if ground truth is NA, any prediction is acceptable
        if field_name == 'max_temp':
            # For max_temp: NA in ground truth means "not recorded because no CRS occurred"
            # Any prediction (including extracting a temperature) should be considered correct
            exact_matches = (
                (pred_numeric == truth_numeric) |  # Exact match
                (pred_numeric.isna() & truth_numeric.isna()) |  # Both NA
                (truth_numeric.isna())  # Ground truth NA = any prediction is acceptable
            )
        else:
            # Standard exact match logic for other numeric fields
            exact_matches = (pred_numeric == truth_numeric) | (
                pred_numeric.isna() & truth_numeric.isna()
            )
        
        exact_match_accuracy = exact_matches.mean()
        
        # For extraction metrics, consider non-NaN values as extracted
        pred_extracted = pred_numeric.notna()
        
        # Special handling for max_temp: ground truth is always "available" 
        # because NA means "not recorded because no CRS" which is still valid ground truth
        # This ensures that predictions are not penalized when truth is NA
        if field_name == 'max_temp':
            truth_available = pd.Series([True] * len(truth_numeric), index=truth_numeric.index)
        else:
            truth_available = truth_numeric.notna()
        
    elif field_name in date_fields:
        # Handle date fields
        pred_dates = pd.to_datetime(predicted, errors='coerce')
        truth_dates = pd.to_datetime(ground_truth, errors='coerce')
        
        # Calculate exact match accuracy for dates
        exact_matches = (pred_dates == truth_dates) | (
            pred_dates.isna() & truth_dates.isna()
        )
        exact_match_accuracy = exact_matches.mean()
        
        # For extraction metrics, consider non-NaT values as extracted
        pred_extracted = pred_dates.notna()
        truth_available = truth_dates.notna()
        
    else:
        # Handle Y/N fields and other text fields
        pred_filled = predicted.fillna('').astype(str).str.strip().str.upper()
        truth_filled = ground_truth.fillna('').astype(str).str.strip().str.upper()
        
        # Calculate exact match accuracy
        exact_matches = (pred_filled == truth_filled)
        exact_match_accuracy = exact_matches.mean()
        
        # Calculate extraction metrics (whether field was extracted vs not)
        pred_extracted = (predicted.notna()) & (predicted != '') & (predicted != 'null')
        truth_available = (ground_truth.notna()) & (ground_truth != '') & (ground_truth != 'null')
    
    # True Positives: Correctly extracted (both have values and match)
    tp = ((pred_extracted) & (truth_available) & (exact_matches)).sum()
    
    # False Positives: Extracted but wrong or ground truth is empty
    fp = ((pred_extracted) & (~exact_matches)).sum()
    
    # False Negatives: Should have been extracted but wasn't
    fn = ((~pred_extracted) & (truth_available)).sum()
    
    # True Negatives: Correctly identified as not available
    tn = ((~pred_extracted) & (~truth_available)).sum()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Extraction rate (how often we attempted to extract when truth was available)
    extraction_rate = pred_extracted.sum() / len(predicted) if len(predicted) > 0 else 0.0
    
    # Coverage (how often we extracted when truth was available)
    coverage = (pred_extracted & truth_available).sum() / truth_available.sum() if truth_available.sum() > 0 else 0.0
    
    # Add additional metrics for specific field types
    additional_metrics = {}
    
    if field_name in numeric_fields:
        # Calculate tolerance-based accuracy for numeric fields
        tolerance = 0
        if field_name == 'max_temp':
            tolerance = 1.0  # Within 1 degree
        elif field_name == 'pressor_use_num':
            tolerance = 0  # Exact match required
        elif field_name == 'total_dose_toci':
            tolerance = 0.5  # Within 0.5 dose
        
        if tolerance > 0:
            pred_numeric = pd.to_numeric(predicted, errors='coerce')
            truth_numeric = pd.to_numeric(ground_truth, errors='coerce')
            
            # Calculate tolerance-based matches
            if field_name == 'max_temp':
                # For max_temp: if ground truth is NA (no CRS), any prediction is acceptable
                tolerance_matches = (
                    (abs(pred_numeric - truth_numeric) <= tolerance) | 
                    (pred_numeric.isna() & truth_numeric.isna()) |
                    (truth_numeric.isna())  # Ground truth NA = any prediction is acceptable
                )
            else:
                # Standard tolerance logic for other numeric fields
                tolerance_matches = (
                    (abs(pred_numeric - truth_numeric) <= tolerance) | 
                    (pred_numeric.isna() & truth_numeric.isna())
                )
            
            tolerance_accuracy = tolerance_matches.mean()
            additional_metrics[f'tolerance_accuracy_{tolerance}'] = tolerance_accuracy
            
        # Calculate mean absolute error for extracted values
        pred_numeric = pd.to_numeric(predicted, errors='coerce')
        truth_numeric = pd.to_numeric(ground_truth, errors='coerce')
        both_available = pred_numeric.notna() & truth_numeric.notna()
        if both_available.any():
            mae = abs(pred_numeric[both_available] - truth_numeric[both_available]).mean()
            additional_metrics['mean_absolute_error'] = mae
        else:
            additional_metrics['mean_absolute_error'] = None
    
    elif field_name in date_fields:
        # Calculate day-level tolerance for dates
        pred_dates = pd.to_datetime(predicted, errors='coerce')
        truth_dates = pd.to_datetime(ground_truth, errors='coerce')
        
        both_available = pred_dates.notna() & truth_dates.notna()
        if both_available.any():
            # Calculate day differences
            day_diffs = abs((pred_dates[both_available] - truth_dates[both_available]).dt.days)
            
            # Tolerance-based accuracy (within 1 day)
            day_tolerance_matches = (day_diffs <= 1) | (pred_dates.isna() & truth_dates.isna())
            day_tolerance_accuracy = day_tolerance_matches.sum() / len(predicted)
            additional_metrics['day_tolerance_accuracy_1'] = day_tolerance_accuracy
            
            # Mean absolute error in days
            mae_days = day_diffs.mean()
            additional_metrics['mean_absolute_error_days'] = mae_days
        else:
            additional_metrics['day_tolerance_accuracy_1'] = exact_match_accuracy
            additional_metrics['mean_absolute_error_days'] = None
    
    elif field_name in yn_fields:
        # For Y/N fields, calculate agreement on positive cases
        pred_yn = predicted.fillna('').astype(str).str.strip().str.upper()
        truth_yn = ground_truth.fillna('').astype(str).str.strip().str.upper()
        
        # Calculate sensitivity and specificity for Y/N classification
        both_available = (pred_yn.isin(['Y', 'N'])) & (truth_yn.isin(['Y', 'N']))
        
        if both_available.any():
            pred_binary = (pred_yn[both_available] == 'Y').astype(int)
            truth_binary = (truth_yn[both_available] == 'Y').astype(int)
            
            if len(np.unique(truth_binary)) > 1:  # If there's variation in ground truth
                sensitivity = recall_score(truth_binary, pred_binary, zero_division=0.0)
                specificity_tn = ((pred_binary == 0) & (truth_binary == 0)).sum()
                specificity_fp = ((pred_binary == 1) & (truth_binary == 0)).sum()
                specificity = specificity_tn / (specificity_tn + specificity_fp) if (specificity_tn + specificity_fp) > 0 else 0.0
                
                additional_metrics['sensitivity'] = sensitivity
                additional_metrics['specificity'] = specificity
            else:
                additional_metrics['sensitivity'] = exact_match_accuracy
                additional_metrics['specificity'] = exact_match_accuracy

    result = {
        'field': field_name,
        'exact_match_accuracy': exact_match_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'extraction_rate': extraction_rate,
        'coverage': coverage,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'total_records': len(predicted),
        'ground_truth_available': truth_available.sum(),
        'predictions_made': pred_extracted.sum()
    }
    
    # Add additional metrics
    result.update(additional_metrics)
    return result


def evaluate_model_against_ground_truth(predictions_file: str, ground_truth_file: str) -> pd.DataFrame:
    """Evaluate a single model's predictions against ground truth"""
    
    # Load data
    predictions = pd.read_csv(predictions_file)
    ground_truth = pd.read_csv(ground_truth_file)
    
    print(f"Evaluating: {Path(predictions_file).stem}")
    print(f"Predictions: {len(predictions)} records")
    print(f"Ground truth: {len(ground_truth)} records")
    
    # Convert MRN to string for consistent matching
    predictions['MRN'] = predictions['MRN'].astype(str)
    ground_truth['MRN'] = ground_truth['MRN'].astype(str)
    
    # Determine merge columns - prioritize MRN + BMT_date matching
    merge_columns = ['MRN']
    if 'bmt_date' in predictions.columns and 'BMT_date' in ground_truth.columns:
        # Standardize BMT_date column names
        ground_truth = ground_truth.rename(columns={'BMT_date': 'bmt_date'})
        merge_columns.append('bmt_date')
        print(f"Merging on: {merge_columns}")
    elif 'admit_date' in predictions.columns and 'admit_date' in ground_truth.columns:
        merge_columns.append('admit_date')
        print(f"Merging on: {merge_columns}")
    else:
        print(f"Merging on: MRN only (date columns not available in both files)")
    
    # Check overlap before merging
    pred_mrns = set(predictions['MRN'].unique())
    truth_mrns = set(ground_truth['MRN'].unique())
    common_mrns = pred_mrns.intersection(truth_mrns)
    
    print(f"Unique MRNs in predictions: {len(pred_mrns)}")
    print(f"Unique MRNs in ground truth: {len(truth_mrns)}")
    print(f"Common MRNs: {len(common_mrns)}")
    
    if len(common_mrns) == 0:
        print("Warning: No common MRNs found between predictions and ground truth")
        return pd.DataFrame()
    
    # Only evaluate predictions that have corresponding ground truth
    # This ensures we only evaluate what the model actually attempted to predict
    merged = predictions.merge(ground_truth, on=merge_columns, suffixes=('_pred', '_truth'), how='inner')
    
    if len(merged) == 0:
        print("Warning: No matching records found between predictions and ground truth after merge")
        return pd.DataFrame()
    
    print(f"Successfully matched records for evaluation: {len(merged)}")
    
    # Show coverage statistics
    coverage_pct = (len(merged) / len(predictions)) * 100 if len(predictions) > 0 else 0
    print(f"Coverage: {coverage_pct:.1f}% of predictions have ground truth")
    
    # Calculate metrics for each field
    field_metrics = []
    
    for field in RESEARCH_FIELDS:
        pred_col = f"{field}_pred"
        truth_col = f"{field}_truth"
        
        if pred_col in merged.columns and truth_col in merged.columns:
            metrics = calculate_field_metrics(merged[pred_col], merged[truth_col], field)
            field_metrics.append(metrics)
        else:
            print(f"Warning: Field {field} not found in data")
    
    return pd.DataFrame(field_metrics)


def calculate_overall_metrics(field_metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall metrics across all fields"""
    
    if len(field_metrics_df) == 0:
        return {}
    
    # Macro averages (average across fields)
    macro_precision = field_metrics_df['precision'].mean()
    macro_recall = field_metrics_df['recall'].mean()
    macro_f1 = field_metrics_df['f1_score'].mean()
    macro_accuracy = field_metrics_df['exact_match_accuracy'].mean()
    
    # Micro averages (sum across fields)
    total_tp = field_metrics_df['true_positives'].sum()
    total_fp = field_metrics_df['false_positives'].sum()
    total_fn = field_metrics_df['false_negatives'].sum()
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Overall extraction rate
    overall_extraction_rate = field_metrics_df['extraction_rate'].mean()
    overall_coverage = field_metrics_df['coverage'].mean()
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'overall_extraction_rate': overall_extraction_rate,
        'overall_coverage': overall_coverage
    }


def check_data_alignment(predictions_dir: str, ground_truth_file: str) -> None:
    """Check alignment between prediction files and ground truth"""
    print("\n" + "="*60)
    print("DATA ALIGNMENT ANALYSIS")
    print("="*60)
    
    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_file)
    truth_mrns = set(ground_truth['MRN'].astype(str))
    print(f"Ground truth contains {len(ground_truth)} records with {len(truth_mrns)} unique MRNs")
    
    # Check each prediction file
    predictions_path = Path(predictions_dir)
    prediction_files = list(predictions_path.glob("research_extraction_*.csv"))
    
    for pred_file in prediction_files:
        model_name = pred_file.stem.replace("research_extraction_", "")
        predictions = pd.read_csv(pred_file)
        pred_mrns = set(predictions['MRN'].astype(str))
        
        common_mrns = pred_mrns.intersection(truth_mrns)
        missing_in_pred = truth_mrns - pred_mrns
        extra_in_pred = pred_mrns - truth_mrns
        
        print(f"\n{model_name}:")
        print(f"  Predictions: {len(predictions)} records, {len(pred_mrns)} unique MRNs")
        print(f"  Common MRNs: {len(common_mrns)} ({len(common_mrns)/len(truth_mrns)*100:.1f}% of ground truth)")
        print(f"  Missing from predictions: {len(missing_in_pred)} MRNs")
        print(f"  Extra in predictions: {len(extra_in_pred)} MRNs")
        
        if missing_in_pred and len(missing_in_pred) <= 10:
            print(f"    Missing MRNs: {sorted(list(missing_in_pred))}")
        elif missing_in_pred:
            print(f"    Missing MRNs (first 10): {sorted(list(missing_in_pred))[:10]}")
            
        if extra_in_pred and len(extra_in_pred) <= 10:
            print(f"    Extra MRNs: {sorted(list(extra_in_pred))}")
        elif extra_in_pred:
            print(f"    Extra MRNs (first 10): {sorted(list(extra_in_pred))[:10]}")


def evaluate_all_models(predictions_dir: str, ground_truth_file: str, output_dir: str = None) -> None:
    """Evaluate all models against ground truth"""
    
    predictions_path = Path(predictions_dir)
    
    if not predictions_path.exists():
        print(f"Predictions directory {predictions_dir} not found")
        return
    
    # Find all prediction files
    prediction_files = list(predictions_path.glob("research_extraction_*.csv"))
    
    if not prediction_files:
        print("No prediction files found")
        return
    
    print(f"Found {len(prediction_files)} prediction files")
    
    # First check data alignment
    check_data_alignment(predictions_dir, ground_truth_file)
    
    print("\n" + "="*80)
    print("RESEARCH MODEL EVALUATION")
    print("="*80)
    
    all_results = []
    detailed_results = {}
    
    for pred_file in prediction_files:
        model_name = pred_file.stem.replace("research_extraction_", "")
        
        try:
            # Calculate field-level metrics
            field_metrics = evaluate_model_against_ground_truth(str(pred_file), ground_truth_file)
            
            if len(field_metrics) > 0:
                # Calculate overall metrics
                overall_metrics = calculate_overall_metrics(field_metrics)
                overall_metrics['model'] = model_name
                all_results.append(overall_metrics)
                
                # Store detailed results
                detailed_results[model_name] = field_metrics
                
                print(f"\n{model_name} - Overall Metrics:")
                print(f"  Macro F1: {overall_metrics['macro_f1']:.3f}")
                print(f"  Micro F1: {overall_metrics['micro_f1']:.3f}")
                print(f"  Macro Accuracy: {overall_metrics['macro_accuracy']:.3f}")
                print(f"  Coverage: {overall_metrics['overall_coverage']:.3f}")
                
                # Show tolerance accuracy for numeric/date fields if available
                tolerance_info = []
                for _, row in field_metrics.iterrows():
                    field = row['field']
                    if field in ['max_temp', 'total_dose_toci']:
                        for col in row.index:
                            if col.startswith('tolerance_accuracy_'):
                                tolerance_val = col.split('_')[-1]
                                tolerance_info.append(f"{field} (±{tolerance_val}): {row[col]:.3f}")
                    elif field in ['fever_onset_date', 'last_fever_date', 'toci_start_date', 'toci_stop_date']:
                        if 'day_tolerance_accuracy_1' in row.index:
                            tolerance_info.append(f"{field} (±1 day): {row['day_tolerance_accuracy_1']:.3f}")
                
                if tolerance_info:
                    print(f"  Tolerance Accuracy: {', '.join(tolerance_info)}")
        
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    if not all_results:
        print("No successful evaluations")
        return
    
    # Save results
    output_path = Path(output_dir) if output_dir else predictions_path
    output_path.mkdir(exist_ok=True)
    
    # Save overall comparison
    overall_df = pd.DataFrame(all_results)
    overall_file = output_path / "research_evaluation_overall_metrics.csv"
    overall_df.to_csv(overall_file, index=False)
    print(f"\nOverall metrics saved to: {overall_file}")
    
    # Save detailed field-level results
    for model_name, field_metrics in detailed_results.items():
        detailed_file = output_path / f"research_evaluation_detailed_{model_name}.csv"
        field_metrics.to_csv(detailed_file, index=False)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("RESEARCH MODEL EVALUATION SUMMARY")
    print("="*80)
    
    # Sort by macro F1 score
    overall_df_sorted = overall_df.sort_values('macro_f1', ascending=False)
    
    print("\nRanking by Macro F1 Score:")
    for _, row in overall_df_sorted.iterrows():
        print(f"{row['model']:<20} | Macro F1: {row['macro_f1']:.3f} | Micro F1: {row['micro_f1']:.3f} | Accuracy: {row['macro_accuracy']:.3f}")


def analyze_field_performance(predictions_dir: str, ground_truth_file: str) -> None:
    """Analyze performance by field across all models"""
    
    predictions_path = Path(predictions_dir)
    prediction_files = list(predictions_path.glob("research_extraction_*.csv"))
    
    if not prediction_files:
        print("No prediction files found")
        return
    
    field_performance = {}
    
    for pred_file in prediction_files:
        model_name = pred_file.stem.replace("research_extraction_", "")
        
        try:
            field_metrics = evaluate_model_against_ground_truth(str(pred_file), ground_truth_file)
            
            for _, row in field_metrics.iterrows():
                field = row['field']
                if field not in field_performance:
                    field_performance[field] = []
                
                perf_data = {
                    'model': model_name,
                    'f1_score': row['f1_score'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'accuracy': row['exact_match_accuracy']
                }
                
                # Add field-specific metrics
                if field in ['max_temp', 'pressor_use_num', 'total_dose_toci']:
                    for col in row.index:
                        if col.startswith('tolerance_accuracy_'):
                            perf_data[col] = row[col]
                        elif col == 'mean_absolute_error':
                            perf_data[col] = row[col]
                elif field in ['fever_onset_date', 'last_fever_date', 'toci_start_date', 'toci_stop_date']:
                    if 'day_tolerance_accuracy_1' in row.index:
                        perf_data['day_tolerance_accuracy_1'] = row['day_tolerance_accuracy_1']
                    if 'mean_absolute_error_days' in row.index:
                        perf_data['mean_absolute_error_days'] = row['mean_absolute_error_days']
                elif field in ['crs_y_n', 'hypotension_y_n', 'hypoxia_y_n', 'high_flow_o2_y_n', 'bipap_or_intubation_y_n', 'neurotox_y_n', 'toci_y_n']:
                    if 'sensitivity' in row.index:
                        perf_data['sensitivity'] = row['sensitivity']
                    if 'specificity' in row.index:
                        perf_data['specificity'] = row['specificity']
                
                field_performance[field].append(perf_data)
        
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    # Print field analysis
    print("\n" + "="*80)
    print("RESEARCH FIELD PERFORMANCE ANALYSIS")
    print("="*80)
    
    for field, performances in field_performance.items():
        if not performances:
            continue
        
        print(f"\n{field.upper()}:")
        
        # Calculate average metrics
        avg_f1 = np.mean([p['f1_score'] for p in performances])
        avg_precision = np.mean([p['precision'] for p in performances])
        avg_recall = np.mean([p['recall'] for p in performances])
        avg_accuracy = np.mean([p['accuracy'] for p in performances])
        
        print(f"  Average across models: F1={avg_f1:.3f}, P={avg_precision:.3f}, R={avg_recall:.3f}, Acc={avg_accuracy:.3f}")
        
        # Show field-specific metrics
        if field in ['max_temp', 'pressor_use_num', 'total_dose_toci']:
            tolerance_key = None
            mae_values = []
            for p in performances:
                for key in p.keys():
                    if key.startswith('tolerance_accuracy_'):
                        tolerance_key = key
                        break
                if 'mean_absolute_error' in p and p['mean_absolute_error'] is not None:
                    mae_values.append(p['mean_absolute_error'])
            
            if tolerance_key:
                avg_tolerance = np.mean([p.get(tolerance_key, 0) for p in performances])
                tolerance_val = tolerance_key.split('_')[-1]
                print(f"  Average tolerance accuracy (±{tolerance_val}): {avg_tolerance:.3f}")
            
            if mae_values:
                avg_mae = np.mean(mae_values)
                unit = "°F" if field == 'max_temp' else "doses" if field == 'total_dose_toci' else "units"
                print(f"  Average MAE: {avg_mae:.2f} {unit}")
        
        elif field in ['fever_onset_date', 'last_fever_date', 'toci_start_date', 'toci_stop_date']:
            day_tolerance_values = [p.get('day_tolerance_accuracy_1', 0) for p in performances if 'day_tolerance_accuracy_1' in p]
            mae_day_values = [p['mean_absolute_error_days'] for p in performances if 'mean_absolute_error_days' in p and p['mean_absolute_error_days'] is not None]
            
            if day_tolerance_values:
                avg_day_tolerance = np.mean(day_tolerance_values)
                print(f"  Average day tolerance accuracy (±1 day): {avg_day_tolerance:.3f}")
            
            if mae_day_values:
                avg_mae_days = np.mean(mae_day_values)
                print(f"  Average MAE: {avg_mae_days:.2f} days")
        
        elif field in ['crs_y_n', 'hypotension_y_n', 'hypoxia_y_n', 'high_flow_o2_y_n', 'bipap_or_intubation_y_n', 'neurotox_y_n', 'toci_y_n']:
            sens_values = [p.get('sensitivity', 0) for p in performances if 'sensitivity' in p]
            spec_values = [p.get('specificity', 0) for p in performances if 'specificity' in p]
            
            if sens_values:
                avg_sensitivity = np.mean(sens_values)
                print(f"  Average sensitivity: {avg_sensitivity:.3f}")
            
            if spec_values:
                avg_specificity = np.mean(spec_values)
                print(f"  Average specificity: {avg_specificity:.3f}")
        
        # Find best model for this field
        best_model = max(performances, key=lambda x: x['f1_score'])
        print(f"  Best model: {best_model['model']} (F1={best_model['f1_score']:.3f})")
        
        # Show all models for this field
        performances_sorted = sorted(performances, key=lambda x: x['f1_score'], reverse=True)
        for perf in performances_sorted:
            base_info = f"    {perf['model']:<15}: F1={perf['f1_score']:.3f}, P={perf['precision']:.3f}, R={perf['recall']:.3f}"
            
            # Add field-specific metrics
            if field in ['max_temp', 'pressor_use_num', 'total_dose_toci']:
                for key in perf.keys():
                    if key.startswith('tolerance_accuracy_'):
                        base_info += f", Tol={perf[key]:.3f}"
                        break
                if 'mean_absolute_error' in perf and perf['mean_absolute_error'] is not None:
                    base_info += f", MAE={perf['mean_absolute_error']:.2f}"
            elif field in ['fever_onset_date', 'last_fever_date', 'toci_start_date', 'toci_stop_date']:
                if 'day_tolerance_accuracy_1' in perf:
                    base_info += f", Day_Tol={perf['day_tolerance_accuracy_1']:.3f}"
                if 'mean_absolute_error_days' in perf and perf['mean_absolute_error_days'] is not None:
                    base_info += f", MAE_days={perf['mean_absolute_error_days']:.2f}"
            elif field in ['crs_y_n', 'hypotension_y_n', 'hypoxia_y_n', 'high_flow_o2_y_n', 'bipap_or_intubation_y_n', 'neurotox_y_n', 'toci_y_n']:
                if 'sensitivity' in perf:
                    base_info += f", Sens={perf['sensitivity']:.3f}"
                if 'specificity' in perf:
                    base_info += f", Spec={perf['specificity']:.3f}"
            
            print(base_info)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate research information extraction results")
    
    parser.add_argument("--predictions-dir", "-p", required=True,
                       help="Directory containing prediction CSV files")
    parser.add_argument("--ground-truth", "-g", required=False,
                       help="Ground truth CSV file (required unless using --compare-only)")
    parser.add_argument("--output-dir", "-o", default=None,
                       help="Output directory for evaluation results")
    parser.add_argument("--field-analysis", action="store_true",
                       help="Perform detailed field-by-field analysis")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only run basic model comparison (no ground truth needed)")
    parser.add_argument("--check-alignment", action="store_true",
                       help="Only check data alignment between predictions and ground truth")
    
    args = parser.parse_args()
    
    try:
        if args.compare_only:
            # Run basic comparison without ground truth
            compare_models_results(args.predictions_dir)
        elif args.check_alignment:
            # Only check data alignment
            if not args.ground_truth:
                print("Error: --ground-truth required for alignment check")
                return
            check_data_alignment(args.predictions_dir, args.ground_truth)
        else:
            # Run full evaluation with ground truth
            if not args.ground_truth:
                print("Error: --ground-truth required for evaluation. Use --compare-only if no ground truth available.")
                return
            
            evaluate_all_models(args.predictions_dir, args.ground_truth, args.output_dir)
            
            if args.field_analysis:
                analyze_field_performance(args.predictions_dir, args.ground_truth)
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()