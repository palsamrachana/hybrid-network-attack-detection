import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from utils.logger import ids_logger

# Import other modules
from models.ml_models import model_manager
from detection.hybrid_detector import DetectionResult


@dataclass
class ExplanationResult:
    """Result of explainability analysis"""
    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray]
    shap_base_value: Optional[float]
    local_explanation: Dict[str, Any]
    global_explanation: Dict[str, Any]
    visualization_data: Dict[str, str]  # Base64 encoded images
    explanation_text: str


class ExplainabilityEngine:
    """Provides explainable AI capabilities using SHAP"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Explainability Engine initialized")
        
        # SHAP explainer
        self.explainer = None
        self.shap_values_cache = {}
        
        # Feature names for interpretation
        self.feature_names = []
        
    def explain_prediction(self, features: np.ndarray, 
                          model_name: str = 'random_forest',
                          feature_names: Optional[List[str]] = None) -> ExplanationResult:
        """
        Explain a single prediction using SHAP
        
        Args:
            features: Feature array for the prediction
            model_name: Name of the model to explain
            feature_names: List of feature names
            
        Returns:
            ExplanationResult with SHAP analysis
        """
        self.logger.info("Starting prediction explanation with SHAP")
        
        try:
            # Load model
            model = self._load_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found or not trained")
            
            # Ensure features are in correct format (2D array)
            X = self._prepare_features_for_shap(features)
            
            # Set feature names
            if feature_names:
                self.feature_names = feature_names
            elif not self.feature_names:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Initialize SHAP explainer if not already done
            if self.explainer is None:
                self._initialize_explainer(model, X)
            
            # Calculate SHAP values with proper error handling
            shap_values = self._calculate_shap_values_safe(X)
            
            # Generate explanations
            feature_importance = self._calculate_feature_importance_safe(shap_values, X)
            local_explanation = self._generate_local_explanation_safe(shap_values, X)
            global_explanation = self._generate_global_explanation_safe(shap_values)
            
            # Create visualizations
            visualization_data = self._create_visualizations_safe(shap_values, X)
            
            # Generate explanation text
            explanation_text = self._generate_explanation_text_safe(
                feature_importance, local_explanation, global_explanation
            )
            
            result = ExplanationResult(
                feature_importance=feature_importance,
                shap_values=shap_values,
                shap_base_value=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else None,
                local_explanation=local_explanation,
                global_explanation=global_explanation,
                visualization_data=visualization_data,
                explanation_text=explanation_text
            )
            
            self.logger.info("Prediction explanation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prediction explanation: {str(e)}")
            return self._create_error_explanation(str(e))
    
    def explain_detection_result(self, detection_result: DetectionResult,
                                features: np.ndarray,
                                feature_names: Optional[List[str]] = None) -> ExplanationResult:
        """
        Explain a complete detection result
        
        Args:
            detection_result: DetectionResult from hybrid detector
            features: Feature array used for detection
            feature_names: List of feature names
            
        Returns:
            ExplanationResult with comprehensive analysis
        """
        self.logger.info("Starting detection result explanation")
        
        try:
            # Explain ML prediction component
            ml_explanation = self.explain_prediction(
                features, 'random_forest', feature_names
            )
            
            # Enhance explanation with detection context
            enhanced_explanation = self._enhance_with_detection_context(
                ml_explanation, detection_result, features
            )
            
            self.logger.info("Detection result explanation completed")
            return enhanced_explanation
            
        except Exception as e:
            self.logger.error(f"Error in detection result explanation: {str(e)}")
            return self._create_error_explanation(str(e))
    
    def _load_model(self, model_name: str):
        """Load a trained model"""
        try:
            # Try to load from saved models directory
            model_path = f"models/saved/{model_name}_model.pkl"
            import joblib
            model = joblib.load(model_path)
            self.logger.info(f"Successfully loaded model {model_name} from {model_path}")
            return model
        except Exception as e:
            self.logger.warning(f"Could not load model {model_name}: {str(e)}")
            return None
    
    def _initialize_explainer(self, model, background_data: np.ndarray):
        """Initialize SHAP explainer with background data"""
        self.logger.debug("Initializing SHAP explainer")
        
        try:
            # Use a subset of background data for efficiency
            if len(background_data) > 100:
                background_sample = background_data[np.random.choice(
                    len(background_data), 100, replace=False
                )]
            else:
                background_sample = background_data
            
            # Initialize TreeExplainer for tree-based models, KernelExplainer for others
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                self.explainer = shap.TreeExplainer(model)
            else:
                # For SVM, use KernelExplainer with background data
                self.explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    background_sample
                )
            
            self.logger.debug("SHAP explainer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing SHAP explainer: {str(e)}")
            # Fallback to a simple explainer
            self.explainer = None
    
    def _prepare_features_for_shap(self, features: np.ndarray) -> np.ndarray:
        """Ensure features are in correct format for SHAP (2D array)"""
        try:
            # Convert to numpy array if not already
            X = np.array(features)
            
            # Ensure 2D array format
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error preparing features for SHAP: {str(e)}")
            # Return a safe default
            return np.zeros((1, 8))  # Default to 8 features
    
    def _calculate_shap_values_safe(self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values with proper error handling"""
        if self.explainer is None:
            # Return dummy SHAP values if explainer not available
            return np.zeros((1, X.shape[1]))
        
        try:
            # Check cache first
            features_hash = hash(X.tobytes())
            if features_hash in self.shap_values_cache:
                return self.shap_values_cache[features_hash]
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle different SHAP output formats safely
            if isinstance(shap_values, list):
                # For multi-class, take values for the predicted class
                if len(shap_values) > 1:
                    shap_values = shap_values[1]  # Assume binary classification, take positive class
                else:
                    shap_values = shap_values[0]
            
            # Ensure shap_values is a numpy array
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)
            
            # Cache the result
            self.shap_values_cache[features_hash] = shap_values
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {str(e)}")
            return np.zeros((1, X.shape[1]))
    
    def _calculate_feature_importance_safe(self, shap_values: np.ndarray, 
                                         X: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from SHAP values with error handling"""
        try:
            if shap_values is None or len(shap_values) == 0:
                return {}
            
            # Ensure shap_values is a proper numpy array
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)
            
            # Handle different SHAP value shapes
            if len(shap_values.shape) == 1:
                # Single prediction
                importance_scores = np.abs(shap_values)
            elif len(shap_values.shape) == 2:
                # Multiple predictions - take mean
                importance_scores = np.abs(shap_values).mean(axis=0)
            else:
                # Fallback
                importance_scores = np.abs(shap_values.flatten())
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, score in enumerate(importance_scores):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                feature_importance[feature_name] = float(score)
            
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def _generate_local_explanation_safe(self, shap_values: np.ndarray, 
                                       X: np.ndarray) -> Dict[str, Any]:
        """Generate local explanation for the specific prediction with error handling"""
        try:
            if shap_values is None or len(shap_values) == 0:
                return {'error': 'No SHAP values available'}
            
            # Ensure shap_values is a proper numpy array
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)
            
            # Get the SHAP values for this specific instance
            if len(shap_values.shape) == 1:
                instance_shap = shap_values
            elif len(shap_values.shape) == 2:
                instance_shap = shap_values[0]
            else:
                instance_shap = shap_values.flatten()
            
            # Ensure X is a proper numpy array
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            
            # Get feature values for this instance
            if len(X.shape) == 1:
                feature_values = X
            elif len(X.shape) == 2:
                feature_values = X[0]
            else:
                feature_values = X.flatten()
            
            # Find most influential features
            abs_shap = np.abs(instance_shap)
            top_indices = np.argsort(abs_shap)[-5:]  # Top 5 features
            
            local_explanation = {
                'top_features': [],
                'feature_contributions': {},
                'prediction_direction': 'increasing' if np.sum(instance_shap) > 0 else 'decreasing'
            }
            
            for idx in top_indices:
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                contribution = float(instance_shap[idx])
                value = float(feature_values[idx]) if idx < len(feature_values) else 0.0
                local_explanation['top_features'].append({
                    'feature': feature_name,
                    'contribution': contribution,
                    'absolute_contribution': float(abs_shap[idx]),
                    'value': value
                })
                local_explanation['feature_contributions'][feature_name] = contribution
            
            return local_explanation
            
        except Exception as e:
            self.logger.error(f"Error generating local explanation: {str(e)}")
            return {'error': f'Error generating local explanation: {str(e)}'}
    
    def _generate_global_explanation_safe(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """Generate global explanation across all features with error handling"""
        try:
            if shap_values is None or len(shap_values) == 0:
                return {'error': 'No SHAP values available'}
            
            # Calculate global feature importance
            global_importance = np.abs(shap_values).mean(axis=0)
            
            global_explanation = {
                'global_feature_importance': {},
                'mean_absolute_shap': float(np.mean(np.abs(shap_values))),
                'shap_std': float(np.std(shap_values))
            }
            
            for i, importance in enumerate(global_importance):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                global_explanation['global_feature_importance'][feature_name] = float(importance)
            
            # Sort by importance
            global_explanation['global_feature_importance'] = dict(sorted(
                global_explanation['global_feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            return global_explanation
            
        except Exception as e:
            self.logger.error(f"Error generating global explanation: {str(e)}")
            return {'error': f'Error generating global explanation: {str(e)}'}
    
    def _create_visualizations_safe(self, shap_values: np.ndarray, 
                                  X: np.ndarray) -> Dict[str, str]:
        """Create SHAP visualization plots with error handling"""
        visualizations = {}
        
        try:
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            
            # Calculate mean absolute SHAP values
            if shap_values is not None and len(shap_values) > 0:
                mean_shap = np.abs(shap_values).mean(axis=0)
                
                # Get top 10 features
                top_indices = np.argsort(mean_shap)[-10:]
                top_features = [self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}" 
                              for i in top_indices]
                top_values = mean_shap[top_indices]
                
                # Create horizontal bar plot
                plt.barh(range(len(top_features)), top_values)
                plt.yticks(range(len(top_features)), top_features)
                plt.xlabel('Mean |SHAP Value|')
                plt.title('Feature Importance (SHAP)')
                plt.tight_layout()
                
                # Convert to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                visualizations['feature_importance'] = img_base64
                
                plt.close()
                
                # Create SHAP summary plot (simplified)
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(shap_values[0])), shap_values[0], alpha=0.6)
                plt.xlabel('Feature Index')
                plt.ylabel('SHAP Value')
                plt.title('SHAP Values for Current Prediction')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                visualizations['shap_values'] = img_base64
                
                plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _generate_explanation_text_safe(self, feature_importance: Dict[str, float],
                                      local_explanation: Dict[str, Any],
                                      global_explanation: Dict[str, Any]) -> str:
        """Generate human-readable explanation text with error handling"""
        try:
            explanation_parts = []
            
            # Local explanation
            if 'top_features' in local_explanation:
                explanation_parts.append("Local Explanation:")
                for feature_info in local_explanation['top_features'][:3]:
                    direction = "increases" if feature_info['contribution'] > 0 else "decreases"
                    explanation_parts.append(
                        f"- {feature_info['feature']}: {direction} risk score by {abs(feature_info['contribution']):.4f} "
                        f"(value: {feature_info['value']:.4f})"
                    )
            
            # Global importance
            if feature_importance:
                top_feature = list(feature_importance.keys())[0]
                explanation_parts.append(f"\nMost Important Feature Overall: {top_feature}")
            
            # Prediction direction
            if 'prediction_direction' in local_explanation:
                explanation_parts.append(
                    f"\nPrediction Direction: {local_explanation['prediction_direction']}"
                )
            
            # Add error message if there are errors
            if 'error' in local_explanation:
                explanation_parts.append(f"\nError: {local_explanation['error']}")
            
            if 'error' in global_explanation:
                explanation_parts.append(f"\nError: {global_explanation['error']}")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating explanation text: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
    def _enhance_with_detection_context(self, explanation: ExplanationResult,
                                      detection_result: DetectionResult,
                                      features: np.ndarray) -> ExplanationResult:
        """Enhance explanation with detection-specific context"""
        
        # Add detection context to explanation text
        detection_context = f"""
        
Detection Context:
- Threat Classification: {detection_result.threat_classification}
- Risk Score: {detection_result.risk_score:.2%}
- ML Confidence: {detection_result.ml_confidence:.2%}
- Rule-based Alerts: {len(detection_result.rule_based_alerts)}
"""
        
        enhanced_text = explanation.explanation_text + detection_context
        
        # Add detection-specific visualizations if needed
        # (This could include plots showing how features relate to specific attack types)
        
        return ExplanationResult(
            feature_importance=explanation.feature_importance,
            shap_values=explanation.shap_values,
            shap_base_value=explanation.shap_base_value,
            local_explanation=explanation.local_explanation,
            global_explanation=explanation.global_explanation,
            visualization_data=explanation.visualization_data,
            explanation_text=enhanced_text
        )
    
    def _create_error_explanation(self, error_msg: str) -> ExplanationResult:
        """Create a fallback explanation when errors occur"""
        return ExplanationResult(
            feature_importance={},
            shap_values=None,
            shap_base_value=None,
            local_explanation={'error': error_msg},
            global_explanation={'error': error_msg},
            visualization_data={'error': error_msg},
            explanation_text=f"Error generating explanation: {error_msg}"
        )
    
    def get_explanation_summary(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """Get a summary of the explanation results"""
        return {
            'feature_importance': dict(list(explanation.feature_importance.items())[:10]),
            'top_contributing_features': explanation.local_explanation.get('top_features', []),
            'prediction_direction': explanation.local_explanation.get('prediction_direction', 'unknown'),
            'explanation_text': explanation.explanation_text,
            'has_visualizations': len(explanation.visualization_data) > 0
        }


# Global instance
explainability_engine = ExplainabilityEngine()