import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Dict, List, Optional
import warnings
from utils.logger import ids_logger


class MLModelTrainer:
    """Trains and manages ML models for intrusion detection"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("ML Model Trainer initialized")
        
        # Initialize models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        self.models = {
            'random_forest': self.rf_model,
            'svm': self.svm_model
        }
        
        # Model performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train ML models on the provided data
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Starting model training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name} model")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                # Feature importance for Random Forest
                if model_name == 'random_forest':
                    self.feature_importance[model_name] = model.feature_importances_
                
                self.logger.info(f"{model_name} training completed. Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Store performance metrics
        self.model_performance = results
        
        return results
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all trained models"""
        return self.model_performance
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Optional[np.ndarray]:
        """Get feature importance from Random Forest model"""
        if model_name in self.feature_importance:
            return self.feature_importance[model_name]
        return None
    
    def predict_single(self, model_name: str, features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Make prediction for a single sample
        
        Args:
            model_name: Name of the model to use
            features: Feature array for the sample
            
        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Reshape features if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence, probabilities
    
    def save_models(self, model_dir: str = "models/saved/") -> None:
        """
        Save trained models to disk
        
        Args:
            model_dir: Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {model_name} model to {model_path}")
    
    def load_models(self, model_dir: str = "models/saved/") -> None:
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory containing saved models
        """
        for model_name in self.models.keys():
            model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                self.logger.info(f"Loaded {model_name} model from {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")
    
    def evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate ensemble performance by combining multiple models
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with ensemble evaluation results
        """
        self.logger.info("Evaluating ensemble performance")
        
        # Get predictions from all models
        model_predictions = {}
        model_probabilities = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict'):
                model_predictions[model_name] = model.predict(X)
                model_probabilities[model_name] = model.predict_proba(X)
        
        # Ensemble prediction (majority vote)
        ensemble_predictions = []
        ensemble_probabilities = []
        
        for i in range(len(X)):
            votes = []
            probs = []
            
            for model_name in self.models.keys():
                if model_name in model_predictions:
                    votes.append(model_predictions[model_name][i])
                    probs.append(model_probabilities[model_name][i])
            
            # Majority vote
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_pred = unique[np.argmax(counts)]
            
            # Average probabilities
            avg_prob = np.mean(probs, axis=0)
            
            ensemble_predictions.append(ensemble_pred)
            ensemble_probabilities.append(avg_prob)
        
        # Calculate ensemble metrics
        ensemble_predictions = np.array(ensemble_predictions)
        ensemble_probabilities = np.array(ensemble_probabilities)
        
        ensemble_accuracy = accuracy_score(y, ensemble_predictions)
        
        ensemble_results = {
            'ensemble_predictions': ensemble_predictions,
            'ensemble_probabilities': ensemble_probabilities,
            'ensemble_accuracy': ensemble_accuracy,
            'individual_predictions': model_predictions,
            'individual_probabilities': model_probabilities
        }
        
        self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        return ensemble_results


class ModelManager:
    """Manages model lifecycle and operations"""
    
    def __init__(self):
        self.logger = ids_logger
        self.trainer = MLModelTrainer()
        self.models_trained = False
        
    def create_sample_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sample training data for demonstration purposes
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        self.logger.info(f"Creating {n_samples} sample training records")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Feature names (8 features)
        feature_names = [
            'duration', 'src_bytes', 'dst_bytes', 'src_packets', 
            'dst_packets', 'protocol', 'flags', 'additional_metric'
        ]
        
        # Generate features
        features = []
        labels = []
        
        for i in range(n_samples):
            # Generate normal traffic features
            if i < n_samples * 0.8:  # 80% normal traffic
                duration = np.random.exponential(10)
                src_bytes = np.random.exponential(1000)
                dst_bytes = np.random.exponential(500)
                src_packets = np.random.exponential(50)
                dst_packets = np.random.exponential(30)
                protocol = 6.0  # TCP
                flags = np.random.choice([1.0, 2.0, 3.0])  # Normal flags
                additional_metric = src_bytes / 1000.0
                label = 0  # Normal
            else:  # 20% attack traffic
                duration = np.random.exponential(60)  # Longer duration
                src_bytes = np.random.exponential(10000)  # More data
                dst_bytes = np.random.exponential(5000)
                src_packets = np.random.exponential(200)  # More packets
                dst_packets = np.random.exponential(100)
                protocol = 6.0  # TCP
                flags = np.random.choice([4.0, 5.0, 6.0])  # Suspicious flags
                additional_metric = src_bytes / 1000.0
                label = 1  # Attack
            
            features.append([
                duration, src_bytes, dst_bytes, src_packets, 
                dst_packets, protocol, flags, additional_metric
            ])
            labels.append(label)
        
        return np.array(features), np.array(labels), feature_names
    
    def ensure_models_trained(self) -> bool:
        """
        Ensure models are trained. If not, train with sample data.
        
        Returns:
            True if models are ready, False otherwise
        """
        if self.models_trained:
            return True
        
        try:
            # Check if saved models exist
            model_dir = "models/saved/"
            rf_model_path = os.path.join(model_dir, "random_forest_model.pkl")
            svm_model_path = os.path.join(model_dir, "svm_model.pkl")
            
            if os.path.exists(rf_model_path) and os.path.exists(svm_model_path):
                # Load existing models
                self.trainer.load_models(model_dir)
                self.models_trained = True
                self.logger.info("Loaded existing trained models")
                return True
            
            # Train with sample data
            self.logger.info("No existing models found, training with sample data")
            X, y, feature_names = self.create_sample_training_data(2000)
            
            # Train models
            training_results = self.trainer.train_models(X, y)
            
            # Save models
            self.trainer.save_models()
            
            # Get feature importance
            importance = self.trainer.get_feature_importance('random_forest')
            if importance is not None:
                feature_importance_dict = dict(zip(feature_names, importance))
                training_results['feature_importance'] = feature_importance_dict
            
            # Log summary
            for model_name, result in training_results.items():
                if 'accuracy' in result:
                    self.logger.info(f"{model_name} - Accuracy: {result['accuracy']:.4f}, "
                                   f"CV: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
            
            self.models_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to ensure models are trained: {str(e)}")
            return False
    
    def train_and_save_models(self, X: np.ndarray, y: np.ndarray, 
                            feature_names: Optional[List[str]] = None) -> Dict:
        """
        Complete training pipeline: train, evaluate, and save models
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Training results and model information
        """
        self.logger.info("Starting complete model training pipeline")
        
        # Train models
        training_results = self.trainer.train_models(X, y)
        
        # Get feature importance
        if feature_names:
            importance = self.trainer.get_feature_importance('random_forest')
            if importance is not None:
                feature_importance_dict = dict(zip(feature_names, importance))
                training_results['feature_importance'] = feature_importance_dict
        
        # Save models
        self.trainer.save_models()
        
        # Log summary
        for model_name, result in training_results.items():
            if 'accuracy' in result:
                self.logger.info(f"{model_name} - Accuracy: {result['accuracy']:.4f}, "
                               f"CV: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        
        self.models_trained = True
        return training_results
    
    def load_and_predict(self, model_dir: str, features: np.ndarray) -> Dict:
        """
        Load models and make predictions
        
        Args:
            model_dir: Directory containing saved models
            features: Feature array for prediction
            
        Returns:
            Dictionary with predictions from all models
        """
        # Ensure models are trained first
        if not self.ensure_models_trained():
            return {'error': 'Models could not be loaded or trained'}
        
        predictions = {}
        
        for model_name in self.trainer.models.keys():
            try:
                pred, conf, probs = self.trainer.predict_single(model_name, features)
                predictions[model_name] = {
                    'prediction': pred,
                    'confidence': float(conf),  # Ensure confidence is a scalar
                    'probabilities': probs.tolist()
                }
            except Exception as e:
                self.logger.error(f"Error predicting with {model_name}: {str(e)}")
                predictions[model_name] = {'error': str(e)}
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get summary of all models and their performance"""
        performance = self.trainer.get_model_performance()
        
        summary = {
            'models_available': list(self.trainer.models.keys()),
            'model_performance': performance,
            'feature_importance': self.trainer.get_feature_importance()
        }
        
        return summary


# Global instance
model_manager = ModelManager()