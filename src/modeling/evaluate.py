import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import json
import sys
sys.path.append('../modeling')
from model import CNNLSTMModel

class ModelEvaluator:
    def __init__(self, data_dir, model_dir, results_dir):
        """
        Initialize the model evaluator.
        
        Args:
            data_dir (str): Directory containing processed data
            model_dir (str): Directory containing model weights
            results_dir (str): Directory to save evaluation results
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load data and model
        self._load_data()
        self._load_model()
    
    def _load_data(self):
        """Load test data."""
        print("Loading test data...")
        self.X_test = np.load(os.path.join(self.data_dir, 'X_test.npy'))
        self.y_test = np.load(os.path.join(self.data_dir, 'y_test.npy'))
        
        # Load dates for reference
        dates_df = pd.read_csv(os.path.join(self.data_dir, 'sequence_dates.csv'))
        self.test_dates = pd.to_datetime(dates_df['date'].iloc[-len(self.y_test):])
    
    def _load_model(self):
        """Load trained model."""
        print("Loading model...")
        input_shape = self.X_test.shape[1:]
        self.model = CNNLSTMModel(input_shape=input_shape)
        
        # Load best weights
        weights_path = os.path.join(self.model_dir, 'callbacks', 'best_model.h5')
        self.model.load_model(weights_path)
    
    def evaluate(self):
        """Evaluate model performance."""
        print("Making predictions...")
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Save metrics
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        metrics_path = os.path.join(self.results_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nEvaluation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'date': self.test_dates,
            'actual': self.y_test.flatten(),
            'predicted': y_pred.flatten()
        })
        
        # Save predictions
        predictions_path = os.path.join(self.results_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        
        # Generate visualizations
        self._generate_visualizations(predictions_df)
        
        return metrics
    
    def _generate_visualizations(self, predictions_df):
        """Generate evaluation visualizations."""
        # Set style
        plt.style.use('seaborn')
        
        # 1. Actual vs Predicted Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions_df['actual'], predictions_df['predicted'], alpha=0.5)
        plt.plot([predictions_df['actual'].min(), predictions_df['actual'].max()],
                [predictions_df['actual'].min(), predictions_df['actual'].max()],
                'r--', lw=2)
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.title('Actual vs Predicted Returns')
        plt.savefig(os.path.join(self.results_dir, 'actual_vs_predicted.png'))
        plt.close()
        
        # 2. Time Series Plot
        plt.figure(figsize=(12, 6))
        plt.plot(predictions_df['date'], predictions_df['actual'], label='Actual', alpha=0.7)
        plt.plot(predictions_df['date'], predictions_df['predicted'], label='Predicted', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.title('Actual vs Predicted Returns Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'returns_time_series.png'))
        plt.close()
        
        # 3. Error Distribution
        plt.figure(figsize=(10, 6))
        errors = predictions_df['predicted'] - predictions_df['actual']
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.savefig(os.path.join(self.results_dir, 'error_distribution.png'))
        plt.close()

def main():
    # Set paths
    data_dir = '../../data/processed'
    model_dir = '../../models'
    results_dir = '../../results'
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        data_dir=data_dir,
        model_dir=model_dir,
        results_dir=results_dir
    )
    
    # Evaluate model
    metrics = evaluator.evaluate()

if __name__ == "__main__":
    main() 
