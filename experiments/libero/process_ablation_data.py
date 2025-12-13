#!/usr/bin/env python3
"""
Script to process ablation experiment data and generate CSV files with success rates.
Processes three ablation conditions:
1. no_depth_with_trace (no depth + yes visual trace)
2. with_depth_no_trace (yes depth + no visual trace)
3. no_depth_no_trace (no depth + no visual trace)

Data is scattered across multiple directories and dates.
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict

# Define the three ablation conditions and their directory name patterns
# Note: Some directories use different naming conventions (include_depth_yes/no instead of with_depth/no_depth)
ABLATION_CONDITIONS = {
    'no_depth_with_trace': ['no_depth_with_trace_noise_level_None', 'include_depth_no_with_trace_noise_level_None'],
    'with_depth_no_trace': ['with_depth_no_trace_noise_level_None', 'include_depth_yes_no_trace_noise_level_None'],
    'no_depth_no_trace': ['no_depth_no_trace_noise_level_None'],
}

# Base directories to search
BASE_DIRS = [
    '/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero/rollouts/2025_12_07',
    '/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero/rollouts/2025_12_09',
    '/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero/rollouts/2025_12_10',
    '/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero/rollouts/2025_12_11',
    '/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero/rollouts/2025_12_12',
    '/scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts/2025_12_07',
    '/scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts/2025_12_08',
    '/scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts/2025_12_09',
    '/scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts/2025_12_10',
]

OUTPUT_DIR = '/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero/tersoo_bryan_experiment_data'


def parse_success_from_filename(filename):
    """Extract success status from filename."""
    match = re.search(r'--success=(True|False)--', filename)
    if match:
        return match.group(1) == 'True'
    return None


def identify_ablation_condition(dir_name):
    """Identify which ablation condition a directory belongs to."""
    for condition, patterns in ABLATION_CONDITIONS.items():
        # Handle both single pattern (string) and multiple patterns (list)
        pattern_list = patterns if isinstance(patterns, list) else [patterns]
        for pattern in pattern_list:
            if pattern in dir_name:
                return condition
    return None


def process_all_directories():
    """
    Process all directories and collect success data.
    
    Returns:
        Dictionary structure: {ablation_condition: {task_type: {task_id: {success: int, total: int}}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'success': 0, 'total': 0})))
    
    for base_dir in BASE_DIRS:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: Directory not found: {base_dir}")
            continue
        
        # Look for model directories
        for model_dir in base_path.glob('models--*'):
            # Determine task type from model directory name
            if 'Object' in model_dir.name:
                task_type = 'object'
            elif 'Spatial' in model_dir.name:
                task_type = 'spatial'
            else:
                continue
            
            # Navigate to task type directory (object or spatial)
            task_type_dir = model_dir / task_type
            if not task_type_dir.exists():
                continue
            
            # Process each task ID directory
            for task_id_dir in task_type_dir.glob('[0-9]'):
                task_id = task_id_dir.name
                
                # Look for ablation condition directories
                for ablation_dir in task_id_dir.iterdir():
                    if not ablation_dir.is_dir():
                        continue
                    
                    ablation_condition = identify_ablation_condition(ablation_dir.name)
                    if ablation_condition is None:
                        continue
                    
                    # Process .mp4 files in the directory, but only count first 50 episodes total per task_id/ablation_condition
                    # Check current count to avoid exceeding 50 across all directories
                    current_total = results[ablation_condition][task_type][task_id]['total']
                    if current_total >= 50:
                        continue  # Already have 50 episodes for this task_id/ablation_condition
                    
                    mp4_files = sorted(ablation_dir.glob('*.mp4'))
                    for file_path in mp4_files:
                        if current_total >= 50:
                            break
                        filename = file_path.name
                        success = parse_success_from_filename(filename)
                        
                        if success is not None:
                            results[ablation_condition][task_type][task_id]['total'] += 1
                            if success:
                                results[ablation_condition][task_type][task_id]['success'] += 1
                            current_total += 1
    
    return results


def calculate_success_rate(success_count, total_count):
    """Calculate success rate as percentage."""
    if total_count == 0:
        return 0.0
    return (success_count / total_count) * 100.0


def generate_csv(results, output_dir):
    """
    Generate CSV files for each ablation condition and task type.
    
    Args:
        results: Dictionary with ablation_condition -> task_type -> task_id -> stats
        output_dir: Directory to write CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate CSV for each ablation condition and task type combination
    for ablation_condition in ABLATION_CONDITIONS.keys():
        for task_type in ['object', 'spatial']:
            if task_type not in results[ablation_condition]:
                continue
            
            # Prepare data for CSV
            rows = []
            total_success = 0
            total_episodes = 0
            
            # Sort task IDs for consistent output
            task_ids = sorted(results[ablation_condition][task_type].keys(), key=int)
            
            # Expected task IDs (0-9)
            expected_task_ids = set(str(i) for i in range(10))
            found_task_ids = set(task_ids)
            missing_task_ids = expected_task_ids - found_task_ids
            incomplete_task_ids = []
            
            # Add rows for each task ID
            for task_id in sorted(expected_task_ids, key=int):
                if task_id in found_task_ids:
                    data = results[ablation_condition][task_type][task_id]
                    success_count = data['success']
                    total_count = data['total']
                    success_rate = calculate_success_rate(success_count, total_count)
                    
                    # Check if incomplete (not 50 episodes)
                    if total_count != 50:
                        incomplete_task_ids.append((task_id, total_count))
                    
                    rows.append({
                        'Task_ID': task_id,
                        'Success_Count': success_count,
                        'Total_Count': total_count,
                        'Success_Rate_Percent': f'{success_rate:.2f}'
                    })
                    
                    total_success += success_count
                    total_episodes += total_count
                else:
                    # Missing task ID - add row with zeros
                    rows.append({
                        'Task_ID': task_id,
                        'Success_Count': 0,
                        'Total_Count': 0,
                        'Success_Rate_Percent': 'N/A'
                    })
            
            # Add overall row
            expected_total = 500  # 10 task IDs × 50 episodes each
            overall_rate = calculate_success_rate(total_success, total_episodes) if total_episodes > 0 else 0.0
            rows.append({
                'Task_ID': 'Overall',
                'Success_Count': total_success,
                'Total_Count': total_episodes,
                'Success_Rate_Percent': f'{overall_rate:.2f}'
            })
            
            # Write to CSV
            filename = f'{ablation_condition}_{task_type}_results.csv'
            output_file = output_path / filename
            
            fieldnames = ['Task_ID', 'Success_Count', 'Total_Count', 'Success_Rate_Percent']
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"Generated {output_file}")
            print(f"  Overall: {total_success}/{total_episodes} (Expected: {expected_total}, Missing: {expected_total - total_episodes} episodes)")
            if missing_task_ids:
                print(f"  ⚠️  MISSING Task IDs: {', '.join(sorted(missing_task_ids, key=int))}")
            if incomplete_task_ids:
                print(f"  ⚠️  INCOMPLETE Task IDs: {', '.join([f'Task {tid} ({count}/50)' for tid, count in incomplete_task_ids])}")
            print()


def main():
    print("Processing ablation experiment data...")
    print("=" * 80)
    
    # Process all directories
    results = process_all_directories()
    
    # Generate CSV files
    print("\nGenerating CSV files...")
    print("=" * 80)
    generate_csv(results, OUTPUT_DIR)
    
    print("\nDone!")
    print(f"All CSV files written to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

