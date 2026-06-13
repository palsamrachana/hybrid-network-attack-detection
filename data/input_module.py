import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional, Tuple
from utils.logger import ids_logger
import os

class NetworkTrafficSimulator:
    """Simulates network traffic data for testing and development"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Network Traffic Simulator initialized")
    
    def generate_sample_data(self, num_records: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic network traffic data
        
        Args:
            num_records: Number of records to generate
            
        Returns:
            DataFrame with network traffic features
        """
        self.logger.info(f"Generating {num_records} sample network traffic records")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        data = []
        
        for i in range(num_records):
            # Basic connection features
            src_ip = self._generate_ip()
            dst_ip = self._generate_ip()
            src_port = random.randint(1000, 65535)
            dst_port = random.choice([80, 443, 22, 21, 25, 53, 3389])
            
            # Protocol selection
            protocol = random.choices(
                ['TCP', 'UDP', 'ICMP'], 
                weights=[0.7, 0.25, 0.05]
            )[0]
            
            # Connection duration
            duration = max(0, np.random.exponential(10))
            
            # Packet counts
            src_bytes = max(0, int(np.random.exponential(1000)))
            dst_bytes = max(0, int(np.random.exponential(500)))
            
            # Flag (connection state)
            flags = random.choice(['SF', 'S0', 'REJ', 'RSTO', 'RSTR'])
            
            # Label (normal or attack)
            is_attack = random.choices([0, 1], weights=[0.8, 0.2])[0]
            
            # Generate attack-specific features if this is an attack
            if is_attack:
                duration = max(duration, 60)  # Attacks tend to last longer
                src_bytes = max(src_bytes, 5000)  # Attacks send more data
                dst_bytes = max(dst_bytes, 2000)
                flags = random.choice(['S0', 'REJ', 'RSTO'])  # Suspicious flags
            
            record = {
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'duration': duration,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'flags': flags,
                'label': 'normal' if is_attack == 0 else 'attack'
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} network traffic records")
        return df
    
    def _generate_ip(self) -> str:
        """Generate a random IP address"""
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"
    
    def load_cicids_sample(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load CICIDS2017 sample data (placeholder implementation)
        
        Args:
            file_path: Path to CICIDS2017 dataset
            
        Returns:
            DataFrame with loaded data
        """
        if file_path and os.path.exists(file_path):
            self.logger.info(f"Loading CICIDS2017 data from {file_path}")
            # In a real implementation, this would load the actual dataset
            # For now, return simulated data with CICIDS-like features
            return self._generate_cicids_like_data()
        else:
            self.logger.warning("CICIDS2017 file not found, generating simulated data")
            return self._generate_cicids_like_data()
    
    def _generate_cicids_like_data(self) -> pd.DataFrame:
        """Generate data that mimics CICIDS2017 structure"""
        self.logger.info("Generating CICIDS2017-like sample data")
        
        # CICIDS2017 has many more features, but we'll simulate a subset
        num_records = 500
        data = []
        
        for i in range(num_records):
            record = {
                'src_ip': self._generate_ip(),
                'dst_ip': self._generate_ip(),
                'src_port': random.randint(1000, 65535),
                'dst_port': random.choice([80, 443, 22, 21, 25, 53]),
                'protocol': random.choice(['TCP', 'UDP', 'ICMP']),
                'duration': max(0, np.random.exponential(10)),
                'src_bytes': max(0, int(np.random.exponential(1000))),
                'dst_bytes': max(0, int(np.random.exponential(500))),
                'flags': random.choice(['SF', 'S0', 'REJ', 'RSTO']),
                
                # Additional CICIDS-like features
                'src_load': max(0, np.random.normal(0.5, 0.2)),
                'dst_load': max(0, np.random.normal(0.3, 0.15)),
                'src_jitter': max(0, np.random.exponential(0.1)),
                'dst_jitter': max(0, np.random.exponential(0.05)),
                'src_packets': max(1, int(np.random.exponential(50))),
                'dst_packets': max(1, int(np.random.exponential(30))),
                
                # Label
                'label': random.choice(['BENIGN', 'DoS', 'PortScan', 'Bot', 'Infiltration'])
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def load_nsl_kdd_sample(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load NSL-KDD sample data (placeholder implementation)
        
        Args:
            file_path: Path to NSL-KDD dataset
            
        Returns:
            DataFrame with loaded data
        """
        if file_path and os.path.exists(file_path):
            self.logger.info(f"Loading NSL-KDD data from {file_path}")
            # In a real implementation, this would load the actual dataset
            return self._generate_nsl_kdd_like_data()
        else:
            self.logger.warning("NSL-KDD file not found, generating simulated data")
            return self._generate_nsl_kdd_like_data()
    
    def _generate_nsl_kdd_like_data(self) -> pd.DataFrame:
        """Generate data that mimics NSL-KDD structure"""
        self.logger.info("Generating NSL-KDD-like sample data")
        
        num_records = 500
        data = []
        
        for i in range(num_records):
            record = {
                'duration': max(0, int(np.random.exponential(10))),
                'protocol_type': random.choice(['tcp', 'udp', 'icmp']),
                'service': random.choice(['http', 'ftp', 'smtp', 'telnet', 'dns']),
                'flag': random.choice(['SF', 'S0', 'REJ', 'RSTO']),
                'src_bytes': max(0, int(np.random.exponential(1000))),
                'dst_bytes': max(0, int(np.random.exponential(500))),
                'land': random.choice([0, 1]),
                'wrong_fragment': max(0, int(np.random.exponential(2))),
                'urgent': max(0, int(np.random.exponential(1))),
                'hot': max(0, int(np.random.exponential(5))),
                'num_failed_logins': max(0, int(np.random.exponential(1))),
                'logged_in': random.choice([0, 1]),
                'num_compromised': max(0, int(np.random.exponential(2))),
                'root_shell': random.choice([0, 1]),
                'su_attempted': random.choice([0, 1, 2]),
                'num_root': max(0, int(np.random.exponential(3))),
                'num_file_creations': max(0, int(np.random.exponential(1))),
                'num_shells': max(0, int(np.random.exponential(1))),
                'num_access_files': max(0, int(np.random.exponential(2))),
                'num_outbound_cmds': 0,  # Always 0 in NSL-KDD
                'is_host_login': random.choice([0, 1]),
                'is_guest_login': random.choice([0, 1]),
                
                # Label
                'label': random.choice(['normal', 'neptune', 'smurf', 'ipsweep', 'portsweep'])
            }
            data.append(record)
        
        return pd.DataFrame(data)


class UserActivityLogger:
    """Handles user activity log collection and processing"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("User Activity Logger initialized")
    
    def simulate_user_activity(self, num_users: int = 10, num_activities: int = 100) -> pd.DataFrame:
        """
        Simulate user activity logs
        
        Args:
            num_users: Number of simulated users
            num_activities: Number of activity records
            
        Returns:
            DataFrame with user activity data
        """
        self.logger.info(f"Simulating user activity for {num_users} users with {num_activities} activities")
        
        users = [f"user{ i + 1}" for i in range(num_users)]
        activities = [
            'login', 'logout', 'file_access', 'database_query', 
            'system_command', 'network_access', 'application_start'
        ]
        
        data = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(num_activities):
            user = random.choice(users)
            activity = random.choice(activities)
            timestamp = base_time + timedelta(minutes=random.randint(0, 1440))
            
            # Generate activity-specific details
            activity_details = self._generate_activity_details(activity)
            
            record = {
                'timestamp': timestamp,
                'user_id': user,
                'activity_type': activity,
                'activity_details': activity_details,
                'ip_address': self._generate_ip(),
                'session_id': f"sess_{random.randint(1000, 9999)}",
                'is_suspicious': random.choices([0, 1], weights=[0.9, 0.1])[0]
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"Generated {len(df)} user activity records")
        return df
    
    def _generate_activity_details(self, activity_type: str) -> str:
        """Generate activity-specific details"""
        if activity_type == 'login':
            return random.choice(['successful', 'failed', 'locked_out'])
        elif activity_type == 'file_access':
            return random.choice(['/home/user/docs/file.txt', '/var/log/system.log', '/etc/passwd'])
        elif activity_type == 'database_query':
            return random.choice(['SELECT', 'INSERT', 'UPDATE', 'DELETE'])
        elif activity_type == 'system_command':
            return random.choice(['ls', 'ps', 'netstat', 'whoami'])
        elif activity_type == 'network_access':
            return random.choice(['http://example.com', 'ftp://files.example.com', 'ssh://server.example.com'])
        elif activity_type == 'application_start':
            return random.choice(['chrome', 'firefox', 'notepad', 'cmd'])
        else:
            return 'unknown'
    
    def _generate_ip(self) -> str:
        """Generate a random IP address"""
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"


# Global instances
traffic_simulator = NetworkTrafficSimulator()
user_activity_logger = UserActivityLogger()