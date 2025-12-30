"""
Sinir Ağı Eğitim Sonuçları için Görselleştirme Modülü
Model performans analizi için kapsamlı grafikler oluşturur.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import sys
import os

sys.path.append(os.path.dirname(__file__))

from data_preprocessing import DataPreprocessor
from neural_network import NeuralNetwork

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_curves(model_path: str = 'models/neural_network.pkl',
                         save_path: str = 'results/training_curves.png'):
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    train_loss = model_data['train_loss_history']
    val_loss = model_data['val_loss_history']
    
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    plt.title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    plt.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_predictions_vs_actual(preprocessor_path: str = 'models/preprocessor.pkl',
                               model_path: str = 'models/neural_network.pkl',
                               save_path: str = 'results/predictions_vs_actual.png'):
    
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor(preprocessor_path)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = NeuralNetwork(
        layer_sizes=model_data['layer_sizes'],
        activation=model_data['activation'],
        learning_rate=model_data['learning_rate']
    )
    model.weights = model_data['weights']
    model.biases = model_data['biases']
    
    df = preprocessor.load_data('data/StudentsPerformance.csv')
    X, y = preprocessor.prepare_data(df, normalize=True, fit=False)
    splits = preprocessor.train_val_test_split(X, y)
    
    X_test = splits['X_test'].T
    y_test = splits['y_test']
    predictions = model.predict(X_test).T
    
    # Denormalize
    predictions_original = preprocessor.denormalize_targets(predictions)
    y_test_original = preprocessor.denormalize_targets(y_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    score_names = ['Math Score', 'Reading Score', 'Writing Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (ax, name, color) in enumerate(zip(axes, score_names, colors)):
        ax.scatter(y_test_original[:, i], predictions_original[:, i], 
                  alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)
        
        min_val = min(y_test_original[:, i].min(), predictions_original[:, i].min())
        max_val = max(y_test_original[:, i].max(), predictions_original[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        
        # calculate R²
        ss_res = np.sum((y_test_original[:, i] - predictions_original[:, i]) ** 2)
        ss_tot = np.sum((y_test_original[:, i] - np.mean(y_test_original[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # calculate RMSE
        rmse = np.sqrt(np.mean((y_test_original[:, i] - predictions_original[:, i]) ** 2))
        
        ax.set_xlabel('Actual Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions vs actual plot saved to {save_path}")
    plt.close()


def plot_error_distribution(preprocessor_path: str = 'models/preprocessor.pkl',
                           model_path: str = 'models/neural_network.pkl',
                           save_path: str = 'results/error_distribution.png'):
    
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor(preprocessor_path)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = NeuralNetwork(
        layer_sizes=model_data['layer_sizes'],
        activation=model_data['activation'],
        learning_rate=model_data['learning_rate']
    )
    model.weights = model_data['weights']
    model.biases = model_data['biases']
    
    df = preprocessor.load_data('data/StudentsPerformance.csv')
    X, y = preprocessor.prepare_data(df, normalize=True, fit=False)
    splits = preprocessor.train_val_test_split(X, y)
    
    X_test = splits['X_test'].T
    y_test = splits['y_test']
    predictions = model.predict(X_test).T
    
    predictions_original = preprocessor.denormalize_targets(predictions)
    y_test_original = preprocessor.denormalize_targets(y_test)
    
    # calculate errors
    errors = y_test_original - predictions_original
    
    # create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    score_names = ['Math Score', 'Reading Score', 'Writing Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (ax, name, color) in enumerate(zip(axes, score_names, colors)):
        # Histogram
        ax.hist(errors[:, i], bins=30, color=color, alpha=0.7, edgecolor='black')
        
        # Add vertical line at 0
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        
        # calculate statistics
        mean_error = np.mean(errors[:, i])
        std_error = np.std(errors[:, i])
        
        ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} Error Distribution\nMean = {mean_error:.2f}, Std = {std_error:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error distribution plot saved to {save_path}")
    plt.close()


def plot_metrics_comparison(results_path: str = 'results/training_results.json',
                           save_path: str = 'results/metrics_comparison.png'):
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    subjects = ['Math', 'Reading', 'Writing']
    
    train_rmse = [results['train_metrics']['math_rmse'],
                  results['train_metrics']['reading_rmse'],
                  results['train_metrics']['writing_rmse']]
    
    val_rmse = [results['val_metrics']['math_rmse'],
                results['val_metrics']['reading_rmse'],
                results['val_metrics']['writing_rmse']]
    
    test_rmse = [results['test_metrics']['math_rmse'],
                 results['test_metrics']['reading_rmse'],
                 results['test_metrics']['writing_rmse']]
    
    train_r2 = [results['train_metrics']['math_r2'],
                results['train_metrics']['reading_r2'],
                results['train_metrics']['writing_r2']]
    
    val_r2 = [results['val_metrics']['math_r2'],
              results['val_metrics']['reading_r2'],
              results['val_metrics']['writing_r2']]
    
    test_r2 = [results['test_metrics']['math_r2'],
               results['test_metrics']['reading_r2'],
               results['test_metrics']['writing_r2']]
    
    # create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(subjects))
    width = 0.25
    
    # RMSE comparison
    ax1.bar(x - width, train_rmse, width, label='Train', color='#3498db', alpha=0.8)
    ax1.bar(x, val_rmse, width, label='Validation', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width, test_rmse, width, label='Test', color='#2ecc71', alpha=0.8)
    
    ax1.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE Comparison Across Subjects', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # R² comparison
    ax2.bar(x - width, train_r2, width, label='Train', color='#3498db', alpha=0.8)
    ax2.bar(x, val_r2, width, label='Validation', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width, test_r2, width, label='Test', color='#2ecc71', alpha=0.8)
    
    ax2.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('R² Score Comparison Across Subjects', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison plot saved to {save_path}")
    plt.close()


def generate_all_visualizations():
    """Generate all visualization plots."""
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    print("\n[1/4] Creating training curves plot...")
    plot_training_curves()
    
    print("\n[2/4] Creating predictions vs actual plot...")
    plot_predictions_vs_actual()
    
    print("\n[3/4] Creating error distribution plot...")
    plot_error_distribution()
    
    print("\n[4/4] Creating metrics comparison plot...")
    plot_metrics_comparison()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print("\nAll plots saved to 'results/' folder:")
    print("  - training_curves.png")
    print("  - predictions_vs_actual.png")
    print("  - error_distribution.png")
    print("  - metrics_comparison.png")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_visualizations()
