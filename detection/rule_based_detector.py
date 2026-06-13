import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re
from utils.logger import ids_logger


class RuleBasedDetector:
    """Implements rule-based detection for known attack patterns"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Rule-Based Detector initialized")
        
        # Initialize attack signatures
        self.attack_signatures = self._load_attack_signatures()
        
        # Initialize anomaly thresholds
        self.anomaly_thresholds = self._load_anomaly_thresholds()
        
        # Connection tracking for stateful analysis
        self.connection_history = {}
        self.suspicious_ips = set()
        
    def _load_attack_signatures(self) -> Dict[str, List[Dict]]:
        """Load predefined attack signatures"""
        signatures = {
            'port_scan': [
                {
                    'name': 'TCP SYN Scan',
                    'description': 'Multiple SYN packets to different ports',
                    'conditions': {
                        'flags': ['S0', 'SF'],
                        'dst_port_count': 10,
                        'time_window': 60
                    }
                },
                {
                    'name': 'UDP Scan',
                    'description': 'Multiple UDP packets to different ports',
                    'conditions': {
                        'protocol': 'UDP',
                        'dst_port_count': 5,
                        'time_window': 30
                    }
                }
            ],
            'dos_attack': [
                {
                    'name': 'High Volume Traffic',
                    'description': 'Excessive traffic from single source',
                    'conditions': {
                        'src_bytes_threshold': 1000000,
                        'connection_count': 100
                    }
                },
                {
                    'name': 'Slowloris',
                    'description': 'Many connections with low data rate',
                    'conditions': {
                        'connection_count': 50,
                        'avg_duration': 300,
                        'avg_src_bytes': 100
                    }
                }
            ],
            'brute_force': [
                {
                    'name': 'SSH Brute Force',
                    'description': 'Multiple failed login attempts',
                    'conditions': {
                        'dst_port': 22,
                        'failed_attempts': 10,
                        'time_window': 300
                    }
                },
                {
                    'name': 'HTTP Brute Force',
                    'description': 'Multiple failed HTTP requests',
                    'conditions': {
                        'dst_port': 80,
                        'failed_attempts': 20,
                        'time_window': 600
                    }
                }
            ],
            'suspicious_activity': [
                {
                    'name': 'Privilege Escalation',
                    'description': 'Suspicious system commands',
                    'conditions': {
                        'command_patterns': ['sudo', 'su', 'passwd', 'useradd']
                    }
                },
                {
                    'name': 'Data Exfiltration',
                    'description': 'Large outbound data transfer',
                    'conditions': {
                        'dst_bytes_threshold': 5000000,
                        'duration': 60
                    }
                }
            ]
        }
        return signatures
    
    def _load_anomaly_thresholds(self) -> Dict[str, float]:
        """Load anomaly detection thresholds"""
        thresholds = {
            'duration_threshold': 3600,  # 1 hour
            'src_bytes_threshold': 10000000,  # 10MB
            'dst_bytes_threshold': 10000000,  # 10MB
            'packet_count_threshold': 10000,
            'connection_frequency_threshold': 100,  # connections per minute
            'unusual_port_threshold': 1000,  # connections to unusual ports
        }
        return thresholds
    
    def detect_attack_signatures(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect known attack patterns in network traffic
        
        Args:
            df: DataFrame with network traffic data
            
        Returns:
            List of detected attacks
        """
        self.logger.info("Starting attack signature detection")
        
        detected_attacks = []
        
        # Check for URG flag attacks (high priority)
        urg_attacks = self._detect_urg_flag_attacks(df)
        detected_attacks.extend(urg_attacks)
        
        # Check for port scans
        port_scan_attacks = self._detect_port_scans(df)
        detected_attacks.extend(port_scan_attacks)
        
        # Check for DoS attacks
        dos_attacks = self._detect_dos_attacks(df)
        detected_attacks.extend(dos_attacks)
        
        # Check for brute force attacks
        brute_force_attacks = self._detect_brute_force(df)
        detected_attacks.extend(brute_force_attacks)
        
        # Check for suspicious activity
        suspicious_activity = self._detect_suspicious_activity(df)
        detected_attacks.extend(suspicious_activity)
        
        self.logger.info(f"Signature detection completed. Found {len(detected_attacks)} potential attacks")
        return detected_attacks
    
    def _detect_urg_flag_attacks(self, df: pd.DataFrame) -> List[Dict]:
        """Detect attacks using URG flag (urgent pointer)"""
        attacks = []
        
        for _, row in df.iterrows():
            if 'flags' in row:
                flags = str(row['flags'])
                if 'U' in flags:  # URG flag detected
                    attacks.append({
                        'attack_type': 'URG_flag_attack',
                        'subtype': 'urgent_pointer_attack',
                        'source_ip': row.get('src_ip', 'unknown'),
                        'destination_ip': row.get('dst_ip', 'unknown'),
                        'flags': flags,
                        'severity': 'high',
                        'confidence': 0.9,
                        'timestamp': datetime.now(),
                        'description': f"URG flag detected - potential attack signature: {flags}"
                    })
        
        return attacks
    
    def _detect_port_scans(self, df: pd.DataFrame) -> List[Dict]:
        """Detect port scanning activity"""
        attacks = []
        
        # Group by source IP
        for src_ip in df['src_ip'].unique():
            src_data = df[df['src_ip'] == src_ip]
            
            # Check for TCP SYN scans
            syn_data = src_data[src_data['flags'].isin(['S0', 'SF'])]
            if len(syn_data) > 0:
                # Check if dst_port exists, if not use a default value
                if 'dst_port' in syn_data.columns:
                    unique_ports = syn_data['dst_port'].nunique()
                    target_ports = sorted(syn_data['dst_port'].unique().tolist())
                else:
                    unique_ports = 1
                    target_ports = [0]  # Default port
                
                if unique_ports >= 10:  # Threshold for port scan
                    attacks.append({
                        'attack_type': 'port_scan',
                        'subtype': 'tcp_syn_scan',
                        'source_ip': src_ip,
                        'target_ports': target_ports,
                        'connection_count': len(syn_data),
                        'severity': 'medium',
                        'confidence': 0.8,
                        'timestamp': datetime.now()
                    })
            
            # Check for UDP scans
            udp_data = src_data[src_data['protocol'] == 'UDP']
            if len(udp_data) > 0:
                # Check if dst_port exists, if not use a default value
                if 'dst_port' in udp_data.columns:
                    unique_ports = udp_data['dst_port'].nunique()
                    target_ports = sorted(udp_data['dst_port'].unique().tolist())
                else:
                    unique_ports = 1
                    target_ports = [0]  # Default port
                
                if unique_ports >= 5:  # Threshold for UDP scan
                    attacks.append({
                        'attack_type': 'port_scan',
                        'subtype': 'udp_scan',
                        'source_ip': src_ip,
                        'target_ports': target_ports,
                        'connection_count': len(udp_data),
                        'severity': 'medium',
                        'confidence': 0.7,
                        'timestamp': datetime.now()
                    })
        
        return attacks
    
    def _detect_dos_attacks(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Denial of Service attacks"""
        attacks = []
        
        # Group by source IP
        for src_ip in df['src_ip'].unique():
            src_data = df[df['src_ip'] == src_ip]
            
            # Check for high volume traffic
            total_bytes = src_data['src_bytes'].sum()
            connection_count = len(src_data)
            
            if total_bytes > self.anomaly_thresholds['src_bytes_threshold'] and connection_count > 100:
                attacks.append({
                    'attack_type': 'dos_attack',
                    'subtype': 'high_volume_traffic',
                    'source_ip': src_ip,
                    'total_bytes': total_bytes,
                    'connection_count': connection_count,
                    'severity': 'high',
                    'confidence': 0.9,
                    'timestamp': datetime.now()
                })
            
            # Check for slowloris-like behavior
            avg_duration = src_data['duration'].mean()
            avg_src_bytes = src_data['src_bytes'].mean()
            
            if connection_count > 50 and avg_duration > 300 and avg_src_bytes < 100:
                attacks.append({
                    'attack_type': 'dos_attack',
                    'subtype': 'slowloris',
                    'source_ip': src_ip,
                    'connection_count': connection_count,
                    'avg_duration': avg_duration,
                    'avg_src_bytes': avg_src_bytes,
                    'severity': 'high',
                    'confidence': 0.85,
                    'timestamp': datetime.now()
                })
        
        return attacks
    
    def _detect_brute_force(self, df: pd.DataFrame) -> List[Dict]:
        """Detect brute force attacks"""
        attacks = []
        
        # Check SSH brute force
        ssh_data = df[df['dst_port'] == 22]
        for src_ip in ssh_data['src_ip'].unique():
            src_ssh_data = ssh_data[ssh_data['src_ip'] == src_ip]
            if len(src_ssh_data) > 10:  # Threshold for brute force
                attacks.append({
                    'attack_type': 'brute_force',
                    'subtype': 'ssh_brute_force',
                    'source_ip': src_ip,
                    'connection_count': len(src_ssh_data),
                    'target_port': 22,
                    'severity': 'high',
                    'confidence': 0.9,
                    'timestamp': datetime.now()
                })
        
        # Check HTTP brute force
        http_data = df[df['dst_port'] == 80]
        for src_ip in http_data['src_ip'].unique():
            src_http_data = http_data[http_data['src_ip'] == src_ip]
            if len(src_http_data) > 20:  # Threshold for brute force
                attacks.append({
                    'attack_type': 'brute_force',
                    'subtype': 'http_brute_force',
                    'source_ip': src_ip,
                    'connection_count': len(src_http_data),
                    'target_port': 80,
                    'severity': 'medium',
                    'confidence': 0.8,
                    'timestamp': datetime.now()
                })
        
        return attacks
    
    def _detect_suspicious_activity(self, df: pd.DataFrame) -> List[Dict]:
        """Detect suspicious network activity"""
        attacks = []
        
        # Check for data exfiltration
        for src_ip in df['src_ip'].unique():
            src_data = df[df['src_ip'] == src_ip]
            total_dst_bytes = src_data['dst_bytes'].sum()
            
            if total_dst_bytes > 5000000:  # 5MB threshold
                attacks.append({
                    'attack_type': 'suspicious_activity',
                    'subtype': 'data_exfiltration',
                    'source_ip': src_ip,
                    'total_dst_bytes': total_dst_bytes,
                    'severity': 'high',
                    'confidence': 0.8,
                    'timestamp': datetime.now()
                })
        
        return attacks
    
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect anomalies using threshold-based rules
        
        Args:
            df: DataFrame with network traffic data
            
        Returns:
            List of detected anomalies
        """
        self.logger.info("Starting anomaly detection")
        
        anomalies = []
        
        # Check individual connection anomalies
        for _, row in df.iterrows():
            anomaly = self._check_connection_anomaly(row)
            if anomaly:
                anomalies.append(anomaly)
        
        # Check aggregate anomalies
        aggregate_anomalies = self._check_aggregate_anomalies(df)
        anomalies.extend(aggregate_anomalies)
        
        self.logger.info(f"Anomaly detection completed. Found {len(anomalies)} anomalies")
        return anomalies
    
    def _check_connection_anomaly(self, row: pd.Series) -> Optional[Dict]:
        """Check if a single connection is anomalous"""
        anomalies = []
        
        # Check duration anomaly
        if row['duration'] > self.anomaly_thresholds['duration_threshold']:
            anomalies.append({
                'anomaly_type': 'duration',
                'value': row['duration'],
                'threshold': self.anomaly_thresholds['duration_threshold'],
                'severity': 'medium'
            })
        
        # Check source bytes anomaly
        if row['src_bytes'] > self.anomaly_thresholds['src_bytes_threshold']:
            anomalies.append({
                'anomaly_type': 'src_bytes',
                'value': row['src_bytes'],
                'threshold': self.anomaly_thresholds['src_bytes_threshold'],
                'severity': 'high'
            })
        
        # Check destination bytes anomaly
        if row['dst_bytes'] > self.anomaly_thresholds['dst_bytes_threshold']:
            anomalies.append({
                'anomaly_type': 'dst_bytes',
                'value': row['dst_bytes'],
                'threshold': self.anomaly_thresholds['dst_bytes_threshold'],
                'severity': 'high'
            })
        
        if anomalies:
            return {
                'source_ip': row['src_ip'],
                'destination_ip': row['dst_ip'],
                'anomalies': anomalies,
                'severity': max([a['severity'] for a in anomalies]),
                'confidence': 0.7,
                'timestamp': datetime.now()
            }
        
        return None
    
    def _check_aggregate_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Check for aggregate anomalies across the dataset"""
        anomalies = []
        
        # Check for unusual connection patterns
        connection_counts = df['src_ip'].value_counts()
        high_frequency_ips = connection_counts[connection_counts > 100].index.tolist()
        
        for ip in high_frequency_ips:
            anomalies.append({
                'anomaly_type': 'high_connection_frequency',
                'source_ip': ip,
                'connection_count': connection_counts[ip],
                'threshold': 100,
                'severity': 'medium',
                'confidence': 0.8,
                'timestamp': datetime.now()
            })
        
        return anomalies
    
    def update_connection_history(self, df: pd.DataFrame) -> None:
        """Update connection history for stateful analysis"""
        for _, row in df.iterrows():
            src_ip = row['src_ip']
            timestamp = datetime.now()
            
            if src_ip not in self.connection_history:
                self.connection_history[src_ip] = []
            
            self.connection_history[src_ip].append({
                'timestamp': timestamp,
                'dst_ip': row['dst_ip'],
                'dst_port': row['dst_port'],
                'protocol': row['protocol'],
                'duration': row['duration']
            })
            
            # Clean old entries (keep last hour)
            cutoff_time = timestamp - timedelta(hours=1)
            self.connection_history[src_ip] = [
                conn for conn in self.connection_history[src_ip] 
                if conn['timestamp'] > cutoff_time
            ]
    
    def get_suspicious_ips(self) -> List[str]:
        """Get list of currently suspicious IPs"""
        return list(self.suspicious_ips)
    
    def add_suspicious_ip(self, ip: str) -> None:
        """Add an IP to the suspicious list"""
        self.suspicious_ips.add(ip)
        self.logger.warning(f"Added {ip} to suspicious IP list")


class ThresholdAnalyzer:
    """Analyzes thresholds and adjusts them based on historical data"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Threshold Analyzer initialized")
        
        # Historical data for threshold adjustment
        self.historical_data = {
            'normal_duration': [],
            'normal_src_bytes': [],
            'normal_dst_bytes': [],
            'normal_connection_counts': []
        }
    
    def update_thresholds(self, df: pd.DataFrame) -> Dict:
        """
        Update thresholds based on current data
        
        Args:
            df: Current dataset
            
        Returns:
            Updated thresholds
        """
        self.logger.info("Updating thresholds based on current data")
        
        # Update historical data
        self.historical_data['normal_duration'].extend(df['duration'].tolist())
        self.historical_data['normal_src_bytes'].extend(df['src_bytes'].tolist())
        self.historical_data['normal_dst_bytes'].extend(df['dst_bytes'].tolist())
        
        # Calculate new thresholds (95th percentile)
        if len(self.historical_data['normal_duration']) > 100:
            new_duration_threshold = np.percentile(self.historical_data['normal_duration'], 95)
        else:
            new_duration_threshold = 3600
        
        if len(self.historical_data['normal_src_bytes']) > 100:
            new_src_bytes_threshold = np.percentile(self.historical_data['normal_src_bytes'], 95)
        else:
            new_src_bytes_threshold = 10000000
        
        if len(self.historical_data['normal_dst_bytes']) > 100:
            new_dst_bytes_threshold = np.percentile(self.historical_data['normal_dst_bytes'], 95)
        else:
            new_dst_bytes_threshold = 10000000
        
        updated_thresholds = {
            'duration_threshold': new_duration_threshold,
            'src_bytes_threshold': new_src_bytes_threshold,
            'dst_bytes_threshold': new_dst_bytes_threshold
        }
        
        self.logger.info("Thresholds updated successfully")
        return updated_thresholds


# Global instances
rule_based_detector = RuleBasedDetector()
threshold_analyzer = ThresholdAnalyzer()