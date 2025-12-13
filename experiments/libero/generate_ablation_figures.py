#!/usr/bin/env python3
"""
Script to generate PNG figures for ablation study results.
Creates publication-quality grid tables and bar charts.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
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

def create_grid_table_figure():
    """Create a grid table figure showing all 4 combinations."""
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
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ablation Study: Success Rates by Depth Tokens and Visual Trace', 
                 fontsize=16, fontweight='bold', y=1.02, color='black')
    
    for idx, task_type in enumerate(['object', 'spatial']):
        ax = axes[idx]
        
        # Prepare data for grid
        data = [
            [results[task_type]['with_depth_with_trace'], results[task_type]['no_depth_with_trace']],
            [results[task_type]['with_depth_no_trace'], results[task_type]['no_depth_no_trace']]
        ]
        
        # Create heatmap with custom colormap
        im = ax.imshow(data, cmap='RdYlGn', vmin=75, vmax=100, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['With Trace', 'No Trace'], fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['With Depth', 'No Depth'], fontsize=12)
        
        # Add text annotations - all in black
        for i in range(2):
            for j in range(2):
                value = data[i][j]
                # Add asterisk for baseline
                label = f'{value:.1f}%'
                if i == 0 and j == 0:
                    label += '*'
                ax.text(j, i, label, ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='black')
        
        # Add title - black text
        task_title = task_type.capitalize()
        ax.set_title(f'{task_title} Tasks', fontsize=14, fontweight='bold', pad=15, color='black')
        
        # Set all axis labels to black
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='black')
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)
        
        # Add colorbar - black text
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Success Rate (%)', rotation=270, labelpad=20, fontsize=11, color='black')
        cbar.ax.tick_params(labelsize=10, colors='black')
    
    # Add legend for baseline
    legend_elem = mpatches.Patch(color='none', label='* Baseline (MolmoAct Paper)')
    fig.legend(handles=[legend_elem], loc='lower center', ncol=1, fontsize=10, frameon=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    output_file = OUTPUT_DIR / 'ablation_grid_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved grid table to: {output_file}")
    plt.close()
    
    # Also create a bar chart comparison
    create_bar_chart_figure(results)
    
    # Create combined figure
    create_combined_figure(results)

def create_bar_chart_figure(results):
    """Create bar charts comparing all conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ablation Study: Success Rate Comparison', fontsize=16, fontweight='bold', y=1.02, color='black')
    
    conditions = ['With Depth\nWith Trace', 'No Depth\nWith Trace', 
                  'With Depth\nNo Trace', 'No Depth\nNo Trace']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for idx, task_type in enumerate(['object', 'spatial']):
        ax = axes[idx]
        
        values = [
            results[task_type]['with_depth_with_trace'],
            results[task_type]['no_depth_with_trace'],
            results[task_type]['with_depth_no_trace'],
            results[task_type]['no_depth_no_trace']
        ]
        
        bars = ax.bar(conditions, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            label = f'{value:.1f}%'
            if bar == bars[0]:  # First bar is baseline
                label += '*'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label,
                   ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
        
        ax.set_ylabel('Success Rate (%)', fontsize=12, color='black')
        ax.set_ylim([70, 100])
        ax.set_title(f'{task_type.capitalize()} Tasks', fontsize=14, fontweight='bold', color='black')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10, colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
    
    # Add legend for baseline
    legend_elem = mpatches.Patch(color='none', label='* Baseline (MolmoAct Paper)')
    fig.legend(handles=[legend_elem], loc='lower center', ncol=1, fontsize=10, frameon=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    output_file = OUTPUT_DIR / 'ablation_bar_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved bar chart to: {output_file}")
    plt.close()

def create_combined_figure(results):
    """Create a single figure with both grid and bar charts."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top row: Grid tables
    for idx, task_type in enumerate(['object', 'spatial']):
        ax = fig.add_subplot(gs[0, idx])
        
        data = [
            [results[task_type]['with_depth_with_trace'], results[task_type]['no_depth_with_trace']],
            [results[task_type]['with_depth_no_trace'], results[task_type]['no_depth_no_trace']]
        ]
        
        im = ax.imshow(data, cmap='RdYlGn', vmin=75, vmax=100, aspect='auto')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['With Trace', 'No Trace'], fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['With Depth', 'No Depth'], fontsize=11)
        
        for i in range(2):
            for j in range(2):
                value = data[i][j]
                label = f'{value:.1f}%'
                if i == 0 and j == 0:
                    label += '*'
                ax.text(j, i, label, ha='center', va='center', 
                       fontsize=14, fontweight='bold', color='black')
        
        ax.set_title(f'{task_type.capitalize()} Tasks', fontsize=13, fontweight='bold', pad=10, color='black')
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)
        ax.tick_params(colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Success Rate (%)', rotation=270, labelpad=18, fontsize=10, color='black')
        cbar.ax.tick_params(labelsize=9, colors='black')
    
    # Bottom row: Bar charts
    conditions = ['With Depth\nWith Trace', 'No Depth\nWith Trace', 
                  'With Depth\nNo Trace', 'No Depth\nNo Trace']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for idx, task_type in enumerate(['object', 'spatial']):
        ax = fig.add_subplot(gs[1, idx])
        
        values = [
            results[task_type]['with_depth_with_trace'],
            results[task_type]['no_depth_with_trace'],
            results[task_type]['with_depth_no_trace'],
            results[task_type]['no_depth_no_trace']
        ]
        
        bars = ax.bar(conditions, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            label = f'{value:.1f}%'
            if bar == bars[0]:
                label += '*'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        
        ax.set_ylabel('Success Rate (%)', fontsize=11, color='black')
        ax.set_ylim([70, 100])
        ax.set_title(f'{task_type.capitalize()} Tasks', fontsize=13, fontweight='bold', color='black')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
    
    fig.suptitle('Ablation Study: Depth Tokens and Visual Trace', 
                 fontsize=16, fontweight='bold', y=0.98, color='black')
    
    # Add legend - black text
    legend_elem = mpatches.Patch(color='none', label='* Baseline (MolmoAct Paper)')
    legend = fig.legend(handles=[legend_elem], loc='lower center', ncol=1, fontsize=10, frameon=False, 
                        bbox_to_anchor=(0.5, 0.01))
    for text in legend.get_texts():
        text.set_color('black')
    
    output_file = OUTPUT_DIR / 'ablation_combined_figure.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved combined figure to: {output_file}")
    plt.close()

def main():
    print("Generating ablation study PNG figures...")
    print("=" * 80)
    
    try:
        create_grid_table_figure()
        print("\nAll figures generated successfully!")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install matplotlib:")
        print("  conda activate molmoact")
        print("  conda install matplotlib -y")
        return
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

