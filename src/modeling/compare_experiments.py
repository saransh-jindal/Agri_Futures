import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_experiment_results(experiments_dir):
    """
    Load results from all experiments in the directory.
    
    Args:
        experiments_dir: Directory containing experiment results
    
    Returns:
        dict: Dictionary of experiment results
    """
    logger.info("Loading experiment results...")
    
    results = {}
    for exp_dir in Path(experiments_dir).glob('*_tuning_*'):
        try:
            with open(exp_dir / 'experiment_results.json', 'r') as f:
                results[exp_dir.name] = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load results from {exp_dir}: {str(e)}")
    
    return results

def analyze_hyperparameters(results):
    """
    Analyze hyperparameter distributions and their impact on performance.
    
    Args:
        results: Dictionary of experiment results
    
    Returns:
        pd.DataFrame: DataFrame with hyperparameter analysis
    """
    logger.info("Analyzing hyperparameters...")
    
    # Extract hyperparameters and performance metrics
    data = []
    for exp_name, exp_results in results.items():
        params = exp_results['best_params']
        data.append({
            'experiment': exp_name,
            'model_type': exp_results['optimization_config']['model_type'],
            'timestamp': exp_results['timestamp'],
            **params
        })
    
    return pd.DataFrame(data)

def plot_hyperparameter_analysis(df, save_dir):
    """
    Create visualizations for hyperparameter analysis.
    
    Args:
        df: DataFrame with hyperparameter analysis
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot hyperparameter distributions by model type
    model_types = df['model_type'].unique()
    for model_type in model_types:
        model_df = df[df['model_type'] == model_type]
        
        # Select numeric columns for plotting
        numeric_cols = model_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
        
        # Create pairplot for numeric hyperparameters
        plt.figure(figsize=(15, 10))
        sns.pairplot(model_df[numeric_cols])
        plt.suptitle(f'Hyperparameter Relationships - {model_type} Model')
        plt.savefig(os.path.join(save_dir, f'{model_type}_hyperparameter_relationships.png'))
        plt.close()
        
        # Create boxplots for each hyperparameter
        plt.figure(figsize=(15, 8))
        model_df[numeric_cols].boxplot()
        plt.title(f'Hyperparameter Distributions - {model_type} Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_type}_hyperparameter_distributions.png'))
        plt.close()

def compare_model_performance(results, save_dir):
    """
    Compare performance metrics across different model types and experiments.
    
    Args:
        results: Dictionary of experiment results
        save_dir: Directory to save visualizations
    """
    logger.info("Comparing model performance...")
    
    # Extract performance metrics
    performance_data = []
    for exp_name, exp_results in results.items():
        if 'performance_metrics' in exp_results:
            metrics = exp_results['performance_metrics']
            performance_data.append({
                'experiment': exp_name,
                'model_type': exp_results['optimization_config']['model_type'],
                'timestamp': exp_results['timestamp'],
                **metrics
            })
    
    if not performance_data:
        logger.warning("No performance metrics found in experiment results")
        return
    
    df = pd.DataFrame(performance_data)
    
    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model_type', y='val_loss', data=df)
    plt.title('Validation Loss Comparison')
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'))
    plt.close()
    
    # Create performance summary table
    summary = df.groupby('model_type').agg({
        'val_loss': ['mean', 'std', 'min'],
        'val_rmse': ['mean', 'std', 'min']
    }).round(4)
    
    summary.to_csv(os.path.join(save_dir, 'performance_summary.csv'))

def generate_comparison_report(results_dir, output_dir):
    """
    Generate a comprehensive comparison report for all experiments.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save comparison report
    """
    logger.info("Generating comparison report...")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join(output_dir, f'comparison_report_{timestamp}')
    os.makedirs(report_dir, exist_ok=True)
    
    # Load and analyze results
    results = load_experiment_results(results_dir)
    if not results:
        logger.error("No experiment results found")
        return
    
    # Analyze hyperparameters
    df = analyze_hyperparameters(results)
    plot_hyperparameter_analysis(df, os.path.join(report_dir, 'hyperparameter_analysis'))
    
    # Compare model performance
    compare_model_performance(results, os.path.join(report_dir, 'performance_analysis'))
    
    # Generate summary report
    summary = {
        'total_experiments': len(results),
        'model_types': df['model_type'].value_counts().to_dict(),
        'date_range': {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max()
        }
    }
    
    with open(os.path.join(report_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Comparison report generated in {report_dir}")

if __name__ == "__main__":
    # Example usage
    results_dir = "experiments"
    output_dir = "reports"
    
    try:
        generate_comparison_report(results_dir, output_dir)
    except Exception as e:
        logger.error(f"Error generating comparison report: {str(e)}")
        raise 
