import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import warnings
from utils.logger import ids_logger


class DataPreprocessor:
    """Handles data preprocessing, feature extraction, and normalization"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Data Preprocessor initialized")
        
        # Initialize scalers and encoders
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Feature columns for different datasets
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
    def preprocess_network_data(self, df: pd.DataFrame, 
                              target_column: str = 'label') -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Preprocess network traffic data for ML models
        
        Args:
            df: Input DataFrame with network traffic data
            target_column: Name of the target/label column
            
        Returns:
            Tuple of (features_df, labels, feature_names)
        """
        self.logger.info("Starting network data preprocessing")
        
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Extract features and labels
        if target_column in data.columns:
            labels = data[target_column].values
            features = data.drop(columns=[target_column])
        else:
            self.logger.warning(f"Target column '{target_column}' not found, using default preprocessing")
            labels = np.zeros(len(data))  # Default labels for unlabeled data
            features = data.copy()
        
        # Identify column types
        self._identify_column_types(features)
        
        # Encode categorical features
        features_encoded = self._encode_categorical_features(features)
        
        # Extract numerical features
        numerical_features = self._extract_numerical_features(features_encoded)
        
        # Normalize features
        normalized_features = self._normalize_features(numerical_features)
        
        # Handle any remaining NaN values
        normalized_features = self._handle_remaining_nans(normalized_features)
        
        self.logger.info(f"Preprocessing completed. Shape: {normalized_features.shape}")
        
        return normalized_features, labels, list(numerical_features.columns)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        self.logger.debug("Handling missing values")
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """Identify categorical and numerical columns"""
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Special handling for common network features
        if 'protocol' in df.columns:
            self.categorical_columns.append('protocol')
        if 'flags' in df.columns:
            self.categorical_columns.append('flags')
        
        # Remove duplicates
        self.categorical_columns = list(set(self.categorical_columns))
        self.numerical_columns = [col for col in self.numerical_columns if col not in self.categorical_columns]
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding"""
        self.logger.debug("Encoding categorical features")
        
        if not self.categorical_columns:
            return df
        
        # One-hot encode categorical features
        encoded_features = pd.get_dummies(df, columns=self.categorical_columns, prefix=self.categorical_columns)
        
        return encoded_features
    
    def _extract_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numerical features for scaling"""
        # Get all numerical columns (including one-hot encoded ones)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure we have some numerical features
        if not numerical_cols:
            # If no numerical features, create a dummy feature
            df['dummy_feature'] = 1.0
            numerical_cols = ['dummy_feature']
        
        return df[numerical_cols].astype(float)
    
    def _normalize_features(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize features using StandardScaler"""
        self.logger.debug("Normalizing features")
        
        # Fit scaler on training data or transform if already fitted
        if hasattr(self.scaler, 'mean_'):
            # Scaler already fitted
            normalized_data = self.scaler.transform(df)
        else:
            # Fit and transform
            normalized_data = self.scaler.fit_transform(df)
        
        return normalized_data
    
    def _handle_remaining_nans(self, data: np.ndarray) -> np.ndarray:
        """Handle any remaining NaN values after preprocessing"""
        if np.isnan(data).any():
            self.logger.warning("Found NaN values after preprocessing, replacing with 0")
            data = np.nan_to_num(data, nan=0.0)
        
        return data
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features from raw data
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature extraction")
        
        # Make a copy
        features = df.copy()
        
        # Add derived features for network data
        if 'duration' in features.columns and 'src_bytes' in features.columns:
            # Calculate data rate
            features['data_rate'] = features['src_bytes'] / (features['duration'] + 1e-6)
        
        if 'src_packets' in features.columns and 'dst_packets' in features.columns:
            # Calculate packet ratio
            features['packet_ratio'] = features['src_packets'] / (features['dst_packets'] + 1)
        
        # Add connection count features (simplified)
        if 'src_ip' in features.columns:
            src_ip_counts = features['src_ip'].value_counts()
            features['src_ip_frequency'] = features['src_ip'].map(src_ip_counts)
        
        if 'dst_ip' in features.columns:
            dst_ip_counts = features['dst_ip'].value_counts()
            features['dst_ip_frequency'] = features['dst_ip'].map(dst_ip_counts)
        
        self.logger.info(f"Feature extraction completed. New shape: {features.shape}")
        return features
    
    def create_feature_dataset(self, df: pd.DataFrame, target_column: str = 'label') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create a complete feature dataset ready for ML models
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        self.logger.info("Creating complete feature dataset")
        
        # Extract features
        features_df = self.extract_features(df)
        
        # Preprocess data
        features, labels, feature_names = self.preprocess_network_data(features_df, target_column)
        
        self.logger.info(f"Feature dataset created. Features shape: {features.shape}")
        return features, labels, feature_names
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of preprocessing steps and parameters"""
        summary = {
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_count': len(self.numerical_columns) if self.numerical_columns else 0
        }
        return summary


class DataEnricher:
    """Handles data enrichment and additional feature engineering"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Data Enricher initialized")
    
    def enrich_network_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich network data with additional computed features
        
        Args:
            df: Input DataFrame with network data
            
        Returns:
            DataFrame with enriched features
        """
        self.logger.info("Starting data enrichment")
        
        enriched_df = df.copy()
        
        # Time-based features (if timestamp available)
        if 'timestamp' in enriched_df.columns:
            enriched_df['hour'] = enriched_df['timestamp'].dt.hour
            enriched_df['day_of_week'] = enriched_df['timestamp'].dt.dayofweek
        
        # Protocol-specific features
        if 'protocol' in enriched_df.columns:
            # Protocol frequency
            protocol_counts = enriched_df['protocol'].value_counts()
            enriched_df['protocol_frequency'] = enriched_df['protocol'].map(protocol_counts)
            
            # Protocol port analysis
            if 'dst_port' in enriched_df.columns:
                enriched_df['is_common_port'] = enriched_df['dst_port'].isin([80, 443, 22, 21, 25, 53]).astype(int)
        
        # Traffic pattern features
        if 'src_bytes' in enriched_df.columns and 'dst_bytes' in enriched_df.columns:
            # Traffic direction analysis
            enriched_df['traffic_asymmetry'] = abs(enriched_df['src_bytes'] - enriched_df['dst_bytes']) / \
                                              (enriched_df['src_bytes'] + enriched_df['dst_bytes'] + 1e-6)
            
            # Total traffic volume
            enriched_df['total_bytes'] = enriched_df['src_bytes'] + enriched_df['dst_bytes']
        
        # Connection state analysis
        if 'flags' in enriched_df.columns:
            # Flag-based features
            enriched_df['has_syn'] = enriched_df['flags'].str.contains('S').astype(int)
            enriched_df['has_ack'] = enriched_df['flags'].str.contains('A').astype(int)
            enriched_df['has_fin'] = enriched_df['flags'].str.contains('F').astype(int)
            enriched_df['has_rst'] = enriched_df['flags'].str.contains('R').astype(int)
        
        self.logger.info(f"Data enrichment completed. New shape: {enriched_df.shape}")
        return enriched_df
    
    def detect_anomalies(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect potential anomalies in the data using statistical methods
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly indicators
        """
        self.logger.info("Starting anomaly detection")
        
        anomaly_df = df.copy()
        
        # Only process numerical columns
        numerical_cols = anomaly_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in ['label']:  # Skip label column
                continue
                
            # Calculate Z-score
            mean_val = anomaly_df[col].mean()
            std_val = anomaly_df[col].std()
            
            if std_val > 0:  # Avoid division by zero
                z_scores = np.abs((anomaly_df[col] - mean_val) / std_val)
                anomaly_df[f'{col}_anomaly'] = (z_scores > threshold).astype(int)
        
        self.logger.info("Anomaly detection completed")
        return anomaly_df


# Global instances
data_preprocessor = DataPreprocessor()
data_enricher = DataEnricher()

def preprocess_live_packet(packet):
    """
    Convert captured packet into feature format compatible with detection_pipeline.
    """
    try:
        # Extract basic packet information
        try:
            src_bytes = len(packet) if hasattr(packet, '__len__') else 0
        except:
            src_bytes = 0

        dst_bytes = 0
        duration = 1

        protocol = "TCP"
        flags = "SYN"

        return {
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "duration": duration,
            "protocol": protocol,
            "flags": flags
        }
        
    except Exception as e:
        ids_logger.error(f"Error preprocessing live packet: {str(e)}")
        # Return default values if processing fails
        return {
            "src_bytes": 0,
            "dst_bytes": 0,
            "duration": 1,
            "protocol": "TCP",
            "flags": "SYN"
        }
