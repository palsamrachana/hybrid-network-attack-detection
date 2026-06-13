import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
from utils.logger import ids_logger

# Import other modules
from models.ml_models import model_manager
from detection.rule_based_detector import rule_based_detector


class ThreatLevel(Enum):
    """Threat level enumeration"""
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class DetectionResult:
    """Result of hybrid detection"""
    threat_classification: str
    risk_score: float
    confidence_score: float
    ml_prediction: str
    ml_confidence: float
    rule_based_alerts: List[Dict]
    combined_confidence: float
    threat_level: ThreatLevel
    explanation: str


class HybridDetector:
    """Main hybrid detection engine combining ML and rule-based approaches"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Hybrid Detector initialized")
        
        # Initialize components
        self.ml_manager = model_manager
        self.rule_detector = rule_based_detector
        
        # Decision thresholds
        self.decision_thresholds = {
            'ml_confidence_threshold': 0.7,
            'risk_score_low': 0.3,
            'risk_score_medium': 0.6,
            'risk_score_high': 0.8,
            'rule_confidence_threshold': 0.5
        }
        
        # Weighting for ensemble decisions
        self.ensemble_weights = {
            'ml_weight': 0.6,
            'rule_weight': 0.4
        }
    
    def analyze_traffic(self, df: pd.DataFrame, 
                       feature_names: Optional[List[str]] = None) -> DetectionResult:
        """
        Perform complete hybrid analysis on network traffic
        
        Args:
            df: DataFrame with network traffic data
            feature_names: List of feature names for ML models
            
        Returns:
            DetectionResult with complete analysis
        """
        self.logger.info("Starting hybrid detection analysis")
        
        # Step 1: ML-based detection
        ml_results = self._perform_ml_detection(df, feature_names)
        
        # Step 2: Rule-based detection
        rule_results = self._perform_rule_detection(df)
        
        # Step 3: Combine results
        combined_result = self._combine_detections(ml_results, rule_results)
        
        self.logger.info(f"Hybrid detection completed. Threat: {combined_result.threat_level.value}")
        return combined_result
    
    def _perform_ml_detection(self, df: pd.DataFrame, 
                            feature_names: Optional[List[str]] = None) -> Dict:
        """Perform ML-based detection using trained models"""
        self.logger.debug("Performing ML-based detection")
        
        try:
            # Get predictions from all models
            ml_predictions = {}
            
            # For each row in the dataframe, make predictions
            for idx, row in df.iterrows():
                # Convert row to feature array
                features = self._extract_features_for_ml(row)
                
                # Get predictions from model manager
                predictions = self.ml_manager.load_and_predict("models/saved/", features)
                
                ml_predictions[idx] = predictions
            
            # Aggregate ML results
            ml_confidence = self._calculate_ml_confidence(ml_predictions)
            ml_prediction = self._determine_ml_prediction(ml_predictions)
            
            return {
                'predictions': ml_predictions,
                'confidence': ml_confidence,
                'prediction': ml_prediction
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML detection: {str(e)}")
            return {
                'predictions': {},
                'confidence': 0.0,
                'prediction': 'unknown'
            }
    
    def _extract_features_for_ml(self, row: pd.Series) -> np.ndarray:
        """Extract and prepare features for ML models"""
        # This is a simplified feature extraction
        # In a real implementation, this would use the preprocessing module
        
        features = []
        
        # Numerical features (8 features total)
        if 'duration' in row:
            features.append(float(row['duration']))
        if 'src_bytes' in row:
            features.append(float(row['src_bytes']))
        if 'dst_bytes' in row:
            features.append(float(row['dst_bytes']))
        if 'src_packets' in row:
            features.append(float(row['src_packets']))
        if 'dst_packets' in row:
            features.append(float(row['dst_packets']))
        
        # Protocol (numeric mapping)
        if 'protocol' in row:
            protocol = str(row['protocol']).upper()
            protocol_map = {'TCP': 6.0, 'UDP': 17.0, 'ICMP': 1.0}
            features.append(protocol_map.get(protocol, 0.0))
        
        # Flags (numeric mapping)
        if 'flags' in row:
            flag = str(row['flags'])
            flag_map = {
                'SF': 1.0,    # SYN-FIN (normal)
                'S': 2.0,     # SYN only
                'A': 3.0,     # ACK only
                'F': 4.0,     # FIN only
                'R': 5.0,     # RST (reset)
                'U': 6.0,     # URG (urgent) - suspicious!
                'SA': 7.0,    # SYN-ACK
                'FA': 8.0,    # FIN-ACK
                'RA': 9.0,    # RST-ACK
                'PU': 10.0,   # PSH-URG - very suspicious!
            }
            features.append(flag_map.get(flag, 0.0))
        
        # Additional metric (use src_bytes as additional metric)
        if 'src_bytes' in row:
            features.append(float(row['src_bytes']) / 1000.0)
        
        # Ensure we have exactly 8 features
        while len(features) < 8:
            features.append(0.0)
        
        return np.array(features[:8])  # Limit to 8 features
    
    def _calculate_ml_confidence(self, predictions: Dict) -> float:
        """Calculate overall ML confidence from multiple models"""
        if not predictions:
            return 0.0
        
        confidences = []
        for pred_dict in predictions.values():
            for model_name, result in pred_dict.items():
                if 'confidence' in result:
                    confidences.append(result['confidence'])
        
        return np.mean(confidences) if confidences else 0.0
    
    def _determine_ml_prediction(self, predictions: Dict) -> str:
        """Determine final ML prediction from multiple models"""
        if not predictions:
            return 'unknown'
        
        predictions_list = []
        for pred_dict in predictions.values():
            for model_name, result in pred_dict.items():
                if 'prediction' in result:
                    predictions_list.append(result['prediction'])
        
        if not predictions_list:
            return 'unknown'
        
        # Return most common prediction
        unique, counts = np.unique(predictions_list, return_counts=True)
        return unique[np.argmax(counts)]
    
    def _perform_rule_detection(self, df: pd.DataFrame) -> Dict:
        """Perform rule-based detection"""
        self.logger.debug("Performing rule-based detection")
        
        try:
            # Detect attack signatures
            attack_signatures = self.rule_detector.detect_attack_signatures(df)
            
            # Detect anomalies
            anomalies = self.rule_detector.detect_anomalies(df)
            
            # Calculate rule-based confidence
            rule_confidence = self._calculate_rule_confidence(attack_signatures, anomalies)
            
            return {
                'attack_signatures': attack_signatures,
                'anomalies': anomalies,
                'confidence': rule_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error in rule-based detection: {str(e)}")
            return {
                'attack_signatures': [],
                'anomalies': [],
                'confidence': 0.0
            }
    
    def _calculate_rule_confidence(self, attacks: List[Dict], anomalies: List[Dict]) -> float:
        """Calculate rule-based confidence from detected patterns"""
        if not attacks and not anomalies:
            return 0.0
        
        # Calculate confidence based on severity and count
        total_confidence = 0.0
        total_weight = 0.0
        
        # Weight attacks more heavily than anomalies
        for attack in attacks:
            severity_weight = {'low': 0.5, 'medium': 0.7, 'high': 1.0}
            weight = severity_weight.get(attack.get('severity', 'low'), 0.5)
            confidence = attack.get('confidence', 0.5)
            
            total_confidence += confidence * weight
            total_weight += weight
        
        for anomaly in anomalies:
            severity_weight = {'low': 0.3, 'medium': 0.5, 'high': 0.8}
            weight = severity_weight.get(anomaly.get('severity', 'low'), 0.3)
            confidence = anomaly.get('confidence', 0.5)
            
            total_confidence += confidence * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _combine_detections(self, ml_results: Dict, rule_results: Dict) -> DetectionResult:
        """Combine ML and rule-based detection results"""
        self.logger.debug("Combining ML and rule-based detections")
        
        # Extract results
        ml_prediction = ml_results.get('prediction', 'unknown')
        ml_confidence = ml_results.get('confidence', 0.0)
        
        rule_alerts = rule_results.get('attack_signatures', []) + rule_results.get('anomalies', [])
        rule_confidence = rule_results.get('confidence', 0.0)
        
        # Apply rule-based overrides for obvious attack patterns
        override_threat_classification = None
        override_risk_score = None
        
        # Check for obvious attack indicators in the input data
        if hasattr(self, '_check_obvious_attacks'):
            # This would check the original DataFrame for obvious attack patterns
            pass
        
        # Calculate combined confidence
        combined_confidence = self._calculate_combined_confidence(ml_confidence, rule_confidence)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(ml_confidence, rule_confidence, rule_alerts)
        
        # Apply rule-based overrides for suspicious patterns
        risk_score, threat_classification, threat_level = self._apply_rule_overrides(
            risk_score, ml_prediction, rule_alerts, ml_confidence
        )
        
        # Generate explanation
        explanation = self._generate_explanation(ml_prediction, ml_confidence, rule_alerts, risk_score)
        
        return DetectionResult(
            threat_classification=threat_classification,
            risk_score=risk_score,
            confidence_score=combined_confidence,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            rule_based_alerts=rule_alerts,
            combined_confidence=combined_confidence,
            threat_level=threat_level,
            explanation=explanation
        )
    
    def _apply_rule_overrides(self, risk_score: float, ml_prediction: str, 
                            rule_alerts: List[Dict], ml_confidence: float) -> Tuple[float, str, ThreatLevel]:
        """
        Apply rule-based overrides for obvious attack patterns
        
        Args:
            risk_score: Current risk score
            ml_prediction: ML prediction
            rule_alerts: List of rule-based alerts
            ml_confidence: ML confidence
            
        Returns:
            Tuple of (updated_risk_score, threat_classification, threat_level)
        """
        # Check for obvious attack indicators in rule alerts
        suspicious_indicators = []
        
        for alert in rule_alerts:
            alert_type = alert.get('attack_type', alert.get('anomaly_type', ''))
            severity = alert.get('severity', 'low')
            
            # High severity alerts automatically increase risk
            if severity == 'high':
                suspicious_indicators.append(f"High severity {alert_type}")
                risk_score = min(risk_score + 0.4, 1.0)
            
            # Specific attack types that should be treated as malicious
            if alert_type in ['URG_flag_attack', 'large_packet_attack', 'port_scan']:
                suspicious_indicators.append(f"Known attack pattern: {alert_type}")
                risk_score = min(risk_score + 0.3, 1.0)
        
        # If we have suspicious indicators, override the classification
        if suspicious_indicators:
            if risk_score >= 0.8:
                return risk_score, "malicious", ThreatLevel.HIGH
            elif risk_score >= 0.6:
                return risk_score, "suspicious", ThreatLevel.MEDIUM
            elif risk_score >= 0.3:
                return risk_score, "potentially_suspicious", ThreatLevel.LOW
        
        # Default determination
        threat_classification, threat_level = self._determine_threat_level(risk_score, ml_prediction, rule_alerts)
        return risk_score, threat_classification, threat_level
    
    def _calculate_combined_confidence(self, ml_confidence: float, rule_confidence: float) -> float:
        """Calculate combined confidence using weighted average"""
        weights = self.ensemble_weights
        return (ml_confidence * weights['ml_weight'] + rule_confidence * weights['rule_weight'])
    
    def _calculate_risk_score(self, ml_confidence: float, rule_confidence: float, 
                            rule_alerts: List[Dict]) -> float:
        """Calculate overall risk score"""
        # Base risk from confidences
        base_risk = (ml_confidence * 0.4) + (rule_confidence * 0.6)
        
        # Adjust based on number and severity of alerts
        if rule_alerts:
            severity_scores = []
            for alert in rule_alerts:
                severity_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
                severity = alert.get('severity', 'low')
                severity_scores.append(severity_map.get(severity, 0.3))
            
            alert_risk = np.mean(severity_scores) if severity_scores else 0.0
            alert_multiplier = 1.0 + (len(rule_alerts) * 0.1)  # More alerts = higher risk
            
            return min(base_risk * alert_multiplier + alert_risk * 0.3, 1.0)
        
        return base_risk
    
    def _determine_threat_level(self, risk_score: float, ml_prediction: str, 
                              rule_alerts: List[Dict]) -> Tuple[str, ThreatLevel]:
        """Determine threat classification and level"""
        thresholds = self.decision_thresholds
        
        # Check for high-confidence rule-based alerts
        high_severity_alerts = [a for a in rule_alerts if a.get('severity') == 'high']
        
        if high_severity_alerts or (risk_score >= thresholds['risk_score_high']):
            return "malicious", ThreatLevel.HIGH
        elif risk_score >= thresholds['risk_score_medium']:
            return "suspicious", ThreatLevel.MEDIUM
        elif risk_score >= thresholds['risk_score_low']:
            return "potentially_suspicious", ThreatLevel.LOW
        else:
            return "normal", ThreatLevel.NORMAL
    
    def _generate_explanation(self, ml_prediction: str, ml_confidence: float, 
                            rule_alerts: List[Dict], risk_score: float) -> str:
        """Generate human-readable explanation of the detection result"""
        explanation_parts = []
        
        # ML explanation
        if ml_confidence > 0.5:
            explanation_parts.append(f"ML models detected {ml_prediction} activity with {ml_confidence:.2%} confidence")
        else:
            explanation_parts.append("ML models found no clear malicious patterns")
        
        # Rule-based explanation
        if rule_alerts:
            alert_types = list(set([alert.get('attack_type', alert.get('anomaly_type', 'unknown')) for alert in rule_alerts]))
            explanation_parts.append(f"Rule-based detection found {len(rule_alerts)} potential issues: {', '.join(alert_types)}")
        else:
            explanation_parts.append("No rule-based anomalies detected")
        
        # Risk explanation
        explanation_parts.append(f"Overall risk score: {risk_score:.2%}")
        
        return ". ".join(explanation_parts)
    
    def get_detection_summary(self, result: DetectionResult) -> Dict:
        """Get a summary of the detection result"""
        return {
            'threat_classification': result.threat_classification,
            'threat_level': result.threat_level.value,
            'risk_score': result.risk_score,
            'confidence_score': result.confidence_score,
            'ml_prediction': result.ml_prediction,
            'ml_confidence': result.ml_confidence,
            'alert_count': len(result.rule_based_alerts),
            'explanation': result.explanation
        }


class DetectionPipeline:
    """Complete detection pipeline with preprocessing and analysis"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Detection Pipeline initialized")
        
        # Import preprocessing module
        from preprocessing.data_preprocessor import data_preprocessor
        
        self.preprocessor = data_preprocessor
        self.hybrid_detector = HybridDetector()
    
    def process_data(self, df: pd.DataFrame, target_column: str = 'label') -> Dict:
        """
        Complete data processing and detection pipeline
        
        Args:
            df: Raw input DataFrame
            target_column: Target column for supervised learning
            
        Returns:
            Complete analysis results
        """
        self.logger.info("Starting complete detection pipeline")
        
        try:
            # Step 1: Feature extraction and preprocessing
            self.logger.info("Step 1: Feature extraction and preprocessing")
            features_df = self.preprocessor.extract_features(df)
            features, labels, feature_names = self.preprocessor.preprocess_network_data(features_df, target_column)
            
            # Step 2: Hybrid detection
            self.logger.info("Step 2: Hybrid detection analysis")
            detection_result = self.hybrid_detector.analyze_traffic(df, feature_names)
            
            # Step 3: Generate summary
            summary = self.hybrid_detector.get_detection_summary(detection_result)
            
            # Combine all results
            results = {
                'preprocessing': {
                    'features_shape': features.shape,
                    'feature_names': feature_names,
                    'labels_shape': labels.shape if labels is not None else None
                },
                'detection': summary,
                'detailed_result': detection_result
            }
            
            self.logger.info("Detection pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in detection pipeline: {str(e)}")
            return {'error': str(e)}


# Global instances
hybrid_detector = HybridDetector()
detection_pipeline = DetectionPipeline()