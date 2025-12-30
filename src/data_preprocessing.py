"""
Öğrenci Performans Veri Kümesi için Veri Ön İşleme Modülü
Verilerin yüklenmesi, kodlanması, normalleştirilmesi ve bölünmesini yönetir.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import pickle

class DataPreprocessor:
    def __init__(self):
        self.feature_encoders = {}
        self.feature_means = None
        self.feature_stds = None
        self.target_means = None
        self.target_stds = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 
                           'lunch', 'test preparation course']
    
        encoded_features = []
        
        for col in categorical_cols:
            if fit:
                unique_values = sorted(df[col].unique())
                self.feature_encoders[col] = {val: idx for idx, val in enumerate(unique_values)}
            
            # create one-hot encoded matrix
            n_categories = len(self.feature_encoders[col])
            one_hot = np.zeros((len(df), n_categories))
            
            for idx, val in enumerate(df[col]):
                cat_idx = self.feature_encoders[col][val]
                one_hot[idx, cat_idx] = 1
            
            encoded_features.append(one_hot)
        
        X = np.concatenate(encoded_features, axis=1)
        
        if fit:
            print(f"\nEncoded feature dimensions: {X.shape}")
            print(f"Total input features: {X.shape[1]}")
        
        return X
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1
        
        X_normalized = (X - self.feature_means) / self.feature_stds
        return X_normalized
    
    def normalize_targets(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit:
            self.target_means = np.mean(y, axis=0)
            self.target_stds = np.std(y, axis=0)
        
        y_normalized = (y - self.target_means) / self.target_stds
        return y_normalized
    
    def denormalize_targets(self, y_normalized: np.ndarray) -> np.ndarray:
        return y_normalized * self.target_stds + self.target_means
    
    def prepare_data(self, df: pd.DataFrame, normalize: bool = True, 
                    fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X = self.encode_categorical_features(df, fit=fit)
        
        target_cols = ['math score', 'reading score', 'writing score']
        y = df[target_cols].values
        
        if normalize:
            X = self.normalize_features(X, fit=fit)
            y = self.normalize_targets(y, fit=fit)
        
        return X, y
    
    def train_val_test_split(self, X: np.ndarray, y: np.ndarray, 
                            train_ratio: float = 0.7, 
                            val_ratio: float = 0.15,
                            random_state: int = 42) -> Dict:
        np.random.seed(random_state)
        
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        # calculate split points
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # create splits
        splits = {
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'X_val': X[val_idx],
            'y_val': y[val_idx],
            'X_test': X[test_idx],
            'y_test': y[test_idx]
        }
        
        print(f"\nData split:")
        print(f"  Training set: {splits['X_train'].shape[0]} samples")
        print(f"  Validation set: {splits['X_val'].shape[0]} samples")
        print(f"  Test set: {splits['X_test'].shape[0]} samples")
        
        return splits
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor parameters."""
        params = {
            'feature_encoders': self.feature_encoders,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'target_means': self.target_means,
            'target_stds': self.target_stds
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor parameters."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.feature_encoders = params['feature_encoders']
        self.feature_means = params['feature_means']
        self.feature_stds = params['feature_stds']
        self.target_means = params['target_means']
        self.target_stds = params['target_stds']
        print(f"Preprocessor loaded from {filepath}")


def main():
    """Example usage of the preprocessor."""
    preprocessor = DataPreprocessor()
    
    df = preprocessor.load_data('data/StudentsPerformance.csv')
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    X, y = preprocessor.prepare_data(df, normalize=True, fit=True)
    splits = preprocessor.train_val_test_split(X, y)
    
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print(f"\nData preprocessing complete!")
    print(f"Input features: {X.shape[1]}")
    print(f"Output targets: {y.shape[1]} (math, reading, writing scores)")


if __name__ == "__main__":
    main()
