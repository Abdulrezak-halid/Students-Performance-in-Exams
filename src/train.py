"""
Öğrenci Performansı Tahmin Modeli için Eğitim Betiği
Veri ön işleme, sinir ağı eğitimi ve değerlendirmeyi entegre eder.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from data_preprocessing import DataPreprocessor
from neural_network import NeuralNetwork
import json


def train_model(data_path: str = 'data/StudentsPerformance.csv',
                hidden_size: int = 64,
                activation: str = 'relu',
                learning_rate: float = 0.01,
                epochs: int = 1000,
                batch_size: int = 32,
                random_state: int = 42):
    print("=" * 80)
    print("STUDENT PERFORMANCE PREDICTION - NEURAL NETWORK TRAINING")
    print("=" * 80)
    
    # ========== 1. DATA PREPROCESSING ==========
    print("\n[STEP 1] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    df = preprocessor.load_data(data_path)
    
    X, y = preprocessor.prepare_data(df, normalize=True, fit=True)
    
    splits = preprocessor.train_val_test_split(
        X, y, 
        train_ratio=0.7, 
        val_ratio=0.15,
        random_state=random_state
    )
    
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # ========== 2. MODEL CREATION ==========
    print("\n[STEP 2] Creating neural network model...")
    input_size = X.shape[1]
    output_size = y.shape[1]
    
    model = NeuralNetwork(
        layer_sizes=[input_size, hidden_size, output_size],
        activation=activation,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    # ========== 3. MODEL TRAINING ==========
    print("\n[STEP 3] Training model...")
    model.fit(
        X_train=splits['X_train'],
        y_train=splits['y_train'],
        X_val=splits['X_val'],
        y_val=splits['y_val'],
        epochs=epochs,
        batch_size=batch_size,
        verbose=100
    )
    
    model.save_model('models/neural_network.pkl')
    
    # ========== 4. MODEL EVALUATION ==========
    print("\n[STEP 4] Evaluating model...")
    
    print("\n--- Training Set Performance ---")
    train_metrics = model.evaluate(splits['X_train'], splits['y_train'])
    print_metrics(train_metrics, preprocessor)
    
    print("\n--- Validation Set Performance ---")
    val_metrics = model.evaluate(splits['X_val'], splits['y_val'])
    print_metrics(val_metrics, preprocessor)
    
    print("\n--- Test Set Performance ---")
    test_metrics = model.evaluate(splits['X_test'], splits['y_test'])
    print_metrics(test_metrics, preprocessor)
    
    # ========== 5. SAVE RESULTS ==========
    print("\n[STEP 5] Saving results...")
    
    results = {
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'activation': activation,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size
        },
        'train_metrics': {
            'math_rmse': float(train_metrics['rmse'][0]),
            'reading_rmse': float(train_metrics['rmse'][1]),
            'writing_rmse': float(train_metrics['rmse'][2]),
            'math_r2': float(train_metrics['r2'][0]),
            'reading_r2': float(train_metrics['r2'][1]),
            'writing_r2': float(train_metrics['r2'][2])
        },
        'val_metrics': {
            'math_rmse': float(val_metrics['rmse'][0]),
            'reading_rmse': float(val_metrics['rmse'][1]),
            'writing_rmse': float(val_metrics['rmse'][2]),
            'math_r2': float(val_metrics['r2'][0]),
            'reading_r2': float(val_metrics['r2'][1]),
            'writing_r2': float(val_metrics['r2'][2])
        },
        'test_metrics': {
            'math_rmse': float(test_metrics['rmse'][0]),
            'reading_rmse': float(test_metrics['rmse'][1]),
            'writing_rmse': float(test_metrics['rmse'][2]),
            'math_r2': float(test_metrics['r2'][0]),
            'reading_r2': float(test_metrics['r2'][1]),
            'writing_r2': float(test_metrics['r2'][2])
        }
    }
    
    with open('results/training_results.json', 'w') as f:
        json.dump(results, indent=4, fp=f)
    
    print("Results saved to results/training_results.json")
    
    # ========== 6. SAMPLE PREDICTIONS ==========
    print("\n[STEP 6] Sample predictions on test set...")
    
    X_test_T = splits['X_test'].T
    y_test_T = splits['y_test'].T
    predictions = model.predict(X_test_T)
    
    predictions_original = preprocessor.denormalize_targets(predictions.T)
    y_test_original = preprocessor.denormalize_targets(splits['y_test'])
    
    print("\nFirst 5 predictions vs actual scores:")
    print("-" * 80)
    print(f"{'Sample':<8} {'Math Pred':<12} {'Math True':<12} {'Read Pred':<12} {'Read True':<12} {'Write Pred':<12} {'Write True':<12}")
    print("-" * 80)
    
    for i in range(min(5, len(predictions_original))):
        print(f"{i+1:<8} "
              f"{predictions_original[i, 0]:>10.2f}  "
              f"{y_test_original[i, 0]:>10.2f}  "
              f"{predictions_original[i, 1]:>10.2f}  "
              f"{y_test_original[i, 1]:>10.2f}  "
              f"{predictions_original[i, 2]:>10.2f}  "
              f"{y_test_original[i, 2]:>10.2f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run 'python src/visualize.py' to generate visualization plots")
    print("  2. Check 'results/' folder for saved metrics and plots")
    print("  3. Review README.md for detailed project documentation")
    print("=" * 80)


def print_metrics(metrics: dict, preprocessor: DataPreprocessor):
    score_names = ['Math', 'Reading', 'Writing']
    
    print(f"\n{'Score':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 40)
    
    for i, name in enumerate(score_names):
        print(f"{name:<10} {metrics['rmse'][i]:>8.4f}  {metrics['mae'][i]:>8.4f}  {metrics['r2'][i]:>8.4f}")
    
    print(f"\nAverage RMSE: {np.mean(metrics['rmse']):.4f}")
    print(f"Average R²: {np.mean(metrics['r2']):.4f}")


if __name__ == "__main__":
    train_model(
        data_path='data/StudentsPerformance.csv',
        hidden_size=64,
        activation='relu',
        learning_rate=0.01,
        epochs=1000,
        batch_size=32,
        random_state=42
    )
