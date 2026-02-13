"""
Evaluation script to fetch metrics from WandB and generate comparison visualizations.
Independent of main.py - run separately after experiments complete.
"""

import os
import sys
import json
import argparse
from typing import List, Dict
import wandb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate and compare experiment runs')
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    parser.add_argument('--run_ids', type=str, required=True, help='JSON list of run IDs to compare')
    parser.add_argument('--wandb_entity', type=str, default='airas', help='WandB entity')
    parser.add_argument('--wandb_project', type=str, default='2026-02-13', help='WandB project')
    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch run data from WandB API.
    
    Returns dict with config, summary metrics, and history.
    """
    api = wandb.Api()
    
    # Find run by ID
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_id})
    
    if not runs:
        # Try by run ID directly
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
        except Exception as e:
            print(f"Warning: Could not fetch run {run_id}: {e}")
            return None
    else:
        run = runs[0]
    
    # Extract data
    data = {
        'run_id': run_id,
        'config': run.config,
        'summary': dict(run.summary),
        'history': run.history().to_dict('records') if hasattr(run, 'history') else []
    }
    
    return data


def export_per_run_metrics(run_data: Dict, results_dir: str):
    """Export per-run metrics to JSON and create visualizations."""
    run_id = run_data['run_id']
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Export metrics JSON
    metrics_file = os.path.join(run_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(run_data['summary'], f, indent=2)
    
    print(f"Exported metrics: {metrics_file}")
    
    # Create per-run figure if we have history
    if run_data['history']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot accuracy over time if available
        history = run_data['history']
        if any('accuracy' in h for h in history):
            steps = [i for i, h in enumerate(history) if 'accuracy' in h]
            accuracies = [h['accuracy'] for h in history if 'accuracy' in h]
            ax.plot(steps, accuracies, marker='o')
            ax.set_xlabel('Step')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Accuracy over Time - {run_id}')
            ax.grid(True, alpha=0.3)
        
        fig_file = os.path.join(run_dir, f'{run_id}_accuracy.pdf')
        plt.savefig(fig_file, bbox_inches='tight')
        plt.close()
        print(f"Exported figure: {fig_file}")


def create_comparison_metrics(all_run_data: List[Dict], results_dir: str):
    """Create aggregated comparison metrics."""
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract metrics by run
    metrics_by_run = {}
    for run_data in all_run_data:
        if run_data:
            run_id = run_data['run_id']
            metrics_by_run[run_id] = run_data['summary']
    
    # Determine primary metric
    primary_metric = 'accuracy'
    
    # Find best proposed and best baseline
    proposed_runs = {k: v for k, v in metrics_by_run.items() if 'proposed' in k}
    baseline_runs = {k: v for k, v in metrics_by_run.items() if 'comparative' in k}
    
    best_proposed = None
    best_proposed_value = -1
    if proposed_runs:
        for run_id, metrics in proposed_runs.items():
            if primary_metric in metrics:
                value = metrics[primary_metric]
                if value > best_proposed_value:
                    best_proposed_value = value
                    best_proposed = run_id
    
    best_baseline = None
    best_baseline_value = -1
    if baseline_runs:
        for run_id, metrics in baseline_runs.items():
            if primary_metric in metrics:
                value = metrics[primary_metric]
                if value > best_baseline_value:
                    best_baseline_value = value
                    best_baseline = run_id
    
    gap = best_proposed_value - best_baseline_value if (best_proposed_value >= 0 and best_baseline_value >= 0) else None
    
    # Create aggregated metrics
    aggregated = {
        'primary_metric': primary_metric,
        'metrics_by_run': metrics_by_run,
        'best_proposed': {
            'run_id': best_proposed,
            'value': best_proposed_value
        } if best_proposed else None,
        'best_baseline': {
            'run_id': best_baseline,
            'value': best_baseline_value
        } if best_baseline else None,
        'gap': gap
    }
    
    # Export
    metrics_file = os.path.join(comparison_dir, 'aggregated_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Exported aggregated metrics: {metrics_file}")


def create_comparison_figures(all_run_data: List[Dict], results_dir: str):
    """Create comparison figures across all runs."""
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Filter valid runs
    valid_runs = [r for r in all_run_data if r is not None]
    
    if not valid_runs:
        print("No valid runs to compare")
        return
    
    # Extract accuracy metrics for bar plot
    run_ids = []
    accuracies = []
    
    for run_data in valid_runs:
        run_id = run_data['run_id']
        if 'accuracy' in run_data['summary']:
            run_ids.append(run_id)
            accuracies.append(run_data['summary']['accuracy'])
    
    if run_ids:
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#2E86AB' if 'proposed' in rid else '#A23B72' for rid in run_ids]
        bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
        
        ax.set_xlabel('Run ID', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Comparison Across Runs', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels(run_ids, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Proposed (XRC-AutoCoT)'),
            Patch(facecolor='#A23B72', label='Baseline (CW-AutoCoT)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        fig_file = os.path.join(comparison_dir, 'comparison_accuracy.pdf')
        plt.savefig(fig_file, bbox_inches='tight')
        plt.close()
        print(f"Exported comparison figure: {fig_file}")
    
    # If we have history data, create overlay plots
    runs_with_history = [r for r in valid_runs if r['history']]
    
    if runs_with_history:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for run_data in runs_with_history:
            run_id = run_data['run_id']
            history = run_data['history']
            
            if any('accuracy' in h for h in history):
                steps = [i for i, h in enumerate(history) if 'accuracy' in h]
                accuracies = [h['accuracy'] for h in history if 'accuracy' in h]
                
                linestyle = '-' if 'proposed' in run_id else '--'
                ax.plot(steps, accuracies, label=run_id, linestyle=linestyle, marker='o', markersize=4)
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Over Time - All Runs', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        fig_file = os.path.join(comparison_dir, 'comparison_accuracy_over_time.pdf')
        plt.savefig(fig_file, bbox_inches='tight')
        plt.close()
        print(f"Exported comparison figure: {fig_file}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating runs: {run_ids}")
    
    # Fetch data from WandB
    all_run_data = []
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        run_data = fetch_run_data(args.wandb_entity, args.wandb_project, run_id)
        
        if run_data:
            # Export per-run metrics
            export_per_run_metrics(run_data, args.results_dir)
            all_run_data.append(run_data)
        else:
            print(f"Warning: Skipping {run_id} (no data found)")
    
    # Create comparison metrics and figures
    print("\nCreating comparison metrics and figures...")
    create_comparison_metrics(all_run_data, args.results_dir)
    create_comparison_figures(all_run_data, args.results_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
