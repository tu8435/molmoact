#!/usr/bin/env python3
"""
Script to generate grid tables and graphs for ablation study results.
Shows all 4 combinations of depth tokens and visual trace for both object and spatial tasks.
"""

import csv
from pathlib import Path

# CSV file directory
CSV_DIR = Path('/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero/tersoo_bryan_experiment_data')
OUTPUT_DIR = CSV_DIR

# MolmoAct paper baseline results (with_depth + with_trace)
PAPER_BASELINE = {
    'object': 95.4,
    'spatial': 87.0
}

def read_csv_results(csv_file):
    """Read CSV file and extract overall success rate."""
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Task_ID'] == 'Overall':
                return float(row['Success_Rate_Percent'])
    return None

def get_color_for_value(value):
    """Get color based on success rate value."""
    if value >= 90:
        return '#2ecc71'  # Green
    elif value >= 85:
        return '#3498db'  # Blue
    elif value >= 80:
        return '#f39c12'  # Orange
    else:
        return '#e74c3c'  # Red

def create_html_grid_table(results):
    """Create an HTML grid table with color coding."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Ablation Study Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
        }
        .container {
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
        }
        .grid-table {
            border-collapse: collapse;
            margin: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            background-color: white;
        }
        .grid-table th {
            background-color: #34495e;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 14px;
            font-weight: bold;
        }
        .grid-table td {
            padding: 20px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            border: 2px solid #ddd;
            min-width: 120px;
        }
        .row-label {
            background-color: #ecf0f1;
            font-weight: bold;
            text-align: right;
            padding-right: 15px;
        }
        .summary-table {
            margin: 30px auto;
            border-collapse: collapse;
            width: 80%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            background-color: white;
        }
        .summary-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .summary-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .summary-table tr:hover {
            background-color: #f5f5f5;
        }
        .baseline {
            font-style: italic;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <h1>Ablation Study: Success Rates by Depth Tokens and Visual Trace</h1>
    <div class="container">
"""
    
    for task_type in ['object', 'spatial']:
        task_title = task_type.capitalize()
        html_content += f"""
        <div>
            <h2 style="text-align: center;">{task_title} Tasks</h2>
            <table class="grid-table">
                <tr>
                    <th></th>
                    <th>With Trace</th>
                    <th>No Trace</th>
                </tr>
                <tr>
                    <td class="row-label">With Depth</td>
                    <td style="background-color: {get_color_for_value(results[task_type]['with_depth_with_trace'])}; color: white;">
                        {results[task_type]['with_depth_with_trace']:.1f}%
                    </td>
                    <td style="background-color: {get_color_for_value(results[task_type]['with_depth_no_trace'])}; color: white;">
                        {results[task_type]['with_depth_no_trace']:.1f}%
                    </td>
                </tr>
                <tr>
                    <td class="row-label">No Depth</td>
                    <td style="background-color: {get_color_for_value(results[task_type]['no_depth_with_trace'])}; color: white;">
                        {results[task_type]['no_depth_with_trace']:.1f}%
                    </td>
                    <td style="background-color: {get_color_for_value(results[task_type]['no_depth_no_trace'])}; color: white;">
                        {results[task_type]['no_depth_no_trace']:.1f}%
                    </td>
                </tr>
            </table>
        </div>
"""
    
    html_content += """
    </div>
    <table class="summary-table">
        <tr>
            <th>Task Type</th>
            <th>Depth Tokens</th>
            <th>Visual Trace</th>
            <th>Success Rate (%)</th>
            <th>Source</th>
        </tr>
"""
    
    for task_type in ['object', 'spatial']:
        html_content += f"""
        <tr>
            <td><strong>{task_type.capitalize()}</strong></td>
            <td>Yes</td>
            <td>Yes</td>
            <td><strong>{results[task_type]['with_depth_with_trace']:.1f}%</strong></td>
            <td class="baseline">MolmoAct Paper (Baseline)</td>
        </tr>
        <tr>
            <td>{task_type.capitalize()}</td>
            <td>No</td>
            <td>Yes</td>
            <td>{results[task_type]['no_depth_with_trace']:.1f}%</td>
            <td>Experiment</td>
        </tr>
        <tr>
            <td>{task_type.capitalize()}</td>
            <td>Yes</td>
            <td>No</td>
            <td>{results[task_type]['with_depth_no_trace']:.1f}%</td>
            <td>Experiment</td>
        </tr>
        <tr>
            <td>{task_type.capitalize()}</td>
            <td>No</td>
            <td>No</td>
            <td>{results[task_type]['no_depth_no_trace']:.1f}%</td>
            <td>Experiment</td>
        </tr>
"""
    
    html_content += """
    </table>
</body>
</html>
"""
    
    output_file = OUTPUT_DIR / 'ablation_results.html'
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Saved HTML grid table to: {output_file}")

def create_text_table(results):
    """Create a text-based grid table."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY: Success Rates by Depth Tokens and Visual Trace")
    print("=" * 80)
    
    for task_type in ['object', 'spatial']:
        print(f"\n{task_type.upper()} TASKS")
        print("-" * 80)
        print(f"{'':<15} {'With Trace':<20} {'No Trace':<20}")
        print("-" * 80)
        print(f"{'With Depth':<15} {results[task_type]['with_depth_with_trace']:>6.1f}%{'':<13} {results[task_type]['with_depth_no_trace']:>6.1f}%")
        print(f"{'No Depth':<15} {results[task_type]['no_depth_with_trace']:>6.1f}%{'':<13} {results[task_type]['no_depth_no_trace']:>6.1f}%")
        print("-" * 80)
    
    print("\n" + "=" * 80)

def create_detailed_table_csv(results):
    """Create a detailed CSV table with all results."""
    output_file = OUTPUT_DIR / 'ablation_summary_table.csv'
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task Type', 'Depth Tokens', 'Visual Trace', 'Success Rate (%)', 'Source'])
        
        for task_type in ['object', 'spatial']:
            writer.writerow([
                task_type.capitalize(),
                'Yes',
                'Yes',
                f"{results[task_type]['with_depth_with_trace']:.1f}",
                'MolmoAct Paper (Baseline)'
            ])
            writer.writerow([
                task_type.capitalize(),
                'No',
                'Yes',
                f"{results[task_type]['no_depth_with_trace']:.1f}",
                'Experiment'
            ])
            writer.writerow([
                task_type.capitalize(),
                'Yes',
                'No',
                f"{results[task_type]['with_depth_no_trace']:.1f}",
                'Experiment'
            ])
            writer.writerow([
                task_type.capitalize(),
                'No',
                'No',
                f"{results[task_type]['no_depth_no_trace']:.1f}",
                'Experiment'
            ])
    
    print(f"Saved detailed table to: {output_file}")

def main():
    print("Generating ablation study graphs and tables...")
    print("=" * 80)
    
    # Read results from CSV files
    results = {
        'object': {
            'with_depth_with_trace': PAPER_BASELINE['object'],
            'no_depth_with_trace': read_csv_results(CSV_DIR / 'no_depth_with_trace_object_results.csv'),
            'with_depth_no_trace': read_csv_results(CSV_DIR / 'with_depth_no_trace_object_results.csv'),
            'no_depth_no_trace': read_csv_results(CSV_DIR / 'no_depth_no_trace_object_results.csv'),
        },
        'spatial': {
            'with_depth_with_trace': PAPER_BASELINE['spatial'],
            'no_depth_with_trace': read_csv_results(CSV_DIR / 'no_depth_with_trace_spatial_results.csv'),
            'with_depth_no_trace': read_csv_results(CSV_DIR / 'with_depth_no_trace_spatial_results.csv'),
            'no_depth_no_trace': read_csv_results(CSV_DIR / 'no_depth_no_trace_spatial_results.csv'),
        }
    }
    
    # Create HTML grid table
    create_html_grid_table(results)
    
    # Create text table
    create_text_table(results)
    
    # Create detailed table CSV
    create_detailed_table_csv(results)
    
    print("\nDone!")
    print(f"All outputs saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

