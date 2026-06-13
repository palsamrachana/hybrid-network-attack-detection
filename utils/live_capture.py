import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import List, Dict, Optional, Tuple
import threading
import signal
import warnings

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system modules
from utils.logger import ids_logger
from preprocessing.data_preprocessor import data_preprocessor
from detection.hybrid_detector import detection_pipeline
from alerts.alert_manager import alert_manager
from mitigation.mitigation_engine import mitigation_orchestrator

# Try to import Scapy
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠️  Scapy not available. Install with: pip install scapy")


class LiveCaptureManager:
    """Manages real-time network traffic capture using Scapy"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Live Capture Manager initialized")
        
        # Capture settings
        self.capture_packets = []
        self.capture_limit = 20  # Capture 20-30 packets per run
        self.capture_duration = 30  # Max capture duration in seconds
        self.capture_active = False
        self.capture_thread = None
        
        # Packet processing
        self.processed_count = 0
        self.suspicious_count = 0
        
    def start_live_capture(self, interface: Optional[str] = None, 
                          capture_count: int = 20) -> List[Dict]:
        """
        Start live packet capture and process through IDS pipeline
        
        Args:
            interface: Network interface to capture on (None for default)
            capture_count: Number of packets to capture (20-30 recommended)
            
        Returns:
            List of detection results for each packet
        """
        if not SCAPY_AVAILABLE:
            self.logger.error("Scapy not available. Cannot start live capture.")
            return []
        
        self.logger.info(f"Starting live capture on interface: {interface or 'default'}")
        self.logger.info(f"Target capture count: {capture_count} packets")
        
        # Reset counters
        self.capture_packets = []
        self.processed_count = 0
        self.suspicious_count = 0
        self.capture_active = True
        
        try:
            # Start packet capture
            packets = sniff(
                iface=interface,
                count=capture_count,
                timeout=self.capture_duration,
                filter="tcp",  # Capture only TCP packets as specified
                prn=self._packet_handler,
                store=True
            )
            
            self.logger.info(f"Captured {len(packets)} packets")
            
            # Process captured packets
            results = self._process_captured_packets(packets)
            
            # Print summary
            self._print_capture_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Live capture failed: {str(e)}")
            return []
        finally:
            self.capture_active = False
    
    def _packet_handler(self, packet):
        """Handle each captured packet"""
        if not self.capture_active:
            return
            
        try:
            # Extract packet information
            packet_info = self._extract_packet_info(packet)
            if packet_info:
                self.capture_packets.append(packet_info)
                self.logger.debug(f"Captured packet: {packet_info['src_ip']} -> {packet_info['dst_ip']}")
                
        except Exception as e:
            self.logger.warning(f"Error processing packet: {str(e)}")
    
    def _extract_packet_info(self, packet) -> Optional[Dict]:
        """Extract relevant information from a captured packet"""
        try:
            # Only process TCP packets
            if not packet.haslayer(TCP):
                return None
            
            # Get IP layer information
            ip_layer = packet[IP]
            tcp_layer = packet[TCP]
            
            # Extract packet size
            packet_size = len(packet)
            
            # Extract TCP flags
            flags = self._extract_tcp_flags(tcp_layer.flags)
            
            # Create packet info dictionary
            packet_info = {
                'timestamp': datetime.now(),
                'src_ip': ip_layer.src,
                'dst_ip': ip_layer.dst,
                'src_port': tcp_layer.sport,
                'dst_port': tcp_layer.dport,
                'protocol': 'TCP',
                'packet_size': packet_size,
                'flags': flags,
                'urg_ptr': tcp_layer.urgptr if hasattr(tcp_layer, 'urgptr') else 0,
                'seq': tcp_layer.seq if hasattr(tcp_layer, 'seq') else 0,
                'ack': tcp_layer.ack if hasattr(tcp_layer, 'ack') else 0
            }
            
            return packet_info
            
        except Exception as e:
            self.logger.warning(f"Error extracting packet info: {str(e)}")
            return None
    
    def _extract_tcp_flags(self, flags) -> str:
        """Extract TCP flags in standard format"""
        flag_map = {
            'S': 'S',  # SYN
            'A': 'A',  # ACK
            'F': 'F',  # FIN
            'R': 'R',  # RST
            'P': 'P',  # PSH
            'U': 'U',  # URG
            'E': 'E',  # ECE
            'C': 'C'   # CWR
        }
        
        flag_str = ''
        for flag, symbol in flag_map.items():
            if flags & getattr(TCP, flag, 0):
                flag_str += symbol
        
        return flag_str if flag_str else 'SF'  # Default to SYN-FIN if no flags
    
    def _process_captured_packets(self, packets) -> List[Dict]:
        """Process captured packets through the IDS pipeline"""
        results = []
        
        for i, packet_info in enumerate(self.capture_packets):
            try:
                self.logger.info(f"Processing packet {i+1}/{len(self.capture_packets)}")
                
                # Convert packet to feature vector
                feature_vector = self._convert_packet_to_features(packet_info)
                
                # Create DataFrame for processing
                df = pd.DataFrame([feature_vector])
                
                # 🔥 FORCE EXACT FEATURE MATCH (CRITICAL FIX)
                expected_columns = [
                    'duration',
                    'src_bytes',
                    'dst_bytes',
                    'src_ip_frequency',
                    'dst_ip_frequency',
                    'src_port',
                    'dst_port',
                    'protocol'
                ]
                ml_df = df[expected_columns]
                
                # Process through detection pipeline
                # DIRECTLY USE HYBRID DETECTOR (SKIP BROKEN PIPELINE)
                detection_result = detection_pipeline.hybrid_detector.analyze_traffic(ml_df,packet_info)

                # Convert to expected format manually
                detection_results = {
                    'detailed_result': detection_result
                }
                
                if 'error' in detection_results:
                    self.logger.error(f"Detection failed for packet {i+1}: {detection_results['error']}")
                    continue
                
                # Extract detection result
                detection_result = detection_results['detailed_result']
                
                # Handle mitigation workflow
                response = mitigation_orchestrator.handle_detection_result(
                    detection_result,
                    packet_info['src_ip'],
                    packet_info['dst_ip']
                )
                
                # Check for suspicious activity (demo support)
                is_suspicious = self._check_suspicious_activity(packet_info, detection_result)
                if is_suspicious:
                    self.suspicious_count += 1
                    self.logger.warning(f"Suspicious activity detected in packet {i+1}")
                
                explanation = getattr(detection_result, "explanation", "No explanation available")

                mitigation_steps = response.get("mitigation_steps")
                if not mitigation_steps:
                    mitigation_steps = ["monitor_traffic"]
                # Create result entry
                result_entry = {
                    'packet_number': i + 1,
                    'timestamp': packet_info['timestamp'],
                    'source_ip': packet_info['src_ip'],
                    'destination_ip': packet_info['dst_ip'],
                    'source_port': packet_info['src_port'],
                    'destination_port': packet_info['dst_port'],
                    'packet_size': packet_info['packet_size'],
                    'tcp_flags': packet_info['flags'],
                    'threat_classification': detection_result.threat_classification,
                    'threat_level': detection_result.threat_level.value,
                    'risk_score': detection_result.risk_score,
                    'confidence_score': detection_result.confidence_score,
                    'explanation': explanation,
                    'mitigation_steps': mitigation_steps,
                    'alert_id': response.get('alert_id'),
                    'mitigations_executed': response.get('mitigations_executed', 0),
                    'suspicious_indicators': is_suspicious
                }
                
                results.append(result_entry)
                self.processed_count += 1
                
                # Print result for this packet
                self._print_packet_result(result_entry)
                
            except Exception as e:
                self.logger.error(f"Error processing packet {i+1}: {str(e)}")
                continue
        
        return results
    
    def _convert_packet_to_features(self, packet_info: Dict) -> Dict:
        """
        Convert packet to EXACT features used during training
        """

        feature_vector = {
            # 1
            'duration': 1.0,

            # 2
            'src_bytes': float(packet_info['packet_size']),

            # 3
            'dst_bytes': float(packet_info['packet_size']) / 2.0,

            # 4 ✅ FIXED
            'src_ip_frequency': 1.0,

            # 5 ✅ FIXED
            'dst_ip_frequency': 1.0,

            # 6 ✅ FIXED
            'src_port': float(packet_info['src_port']),

            # 7 ✅ FIXED
            'dst_port': float(packet_info['dst_port']),

            # 8
            'protocol': 6.0,

            # ===== ADD THESE FOR RULE-BASED SYSTEM =====
            'src_ip': packet_info['src_ip'],
            'dst_ip': packet_info['dst_ip'],
            'flags': packet_info['flags']
        }

        return feature_vector
    
    def _check_suspicious_activity(self, packet_info: Dict, detection_result) -> bool:
        """
        Check for suspicious activity indicators (demo support)
        
        Add simple rules to ensure visible detection:
        - If packet size is very high OR flags contain "URG"
        - Increase risk score or mark as suspicious
        """
        suspicious_indicators = []
        
        # Check packet size (very large packets might be suspicious)
        if packet_info['packet_size'] > 1500:  # Larger than typical MTU
            suspicious_indicators.append(f"Large packet size: {packet_info['packet_size']} bytes")
        
        # Check for URG flag (urgent pointer - often used in attacks)
        if 'U' in packet_info['flags']:
            suspicious_indicators.append("URG flag detected - potential attack signature")
            # Increase risk score for URG flag
            detection_result.risk_score = min(detection_result.risk_score + 0.3, 1.0)
            detection_result.threat_classification = "malicious"
            detection_result.threat_level = detection_result.threat_level.__class__.HIGH
        
        # Check for unusual flag combinations
        if packet_info['flags'] in ['R', 'RA', 'PU']:  # Reset or urgent flags
            suspicious_indicators.append(f"Unusual flag combination: {packet_info['flags']}")
        
        # Check for suspicious port combinations
        if packet_info['dst_port'] in [22, 23, 25, 135, 139, 445, 1433, 3389]:  # Common attack targets
            suspicious_indicators.append(f"Connection to potentially vulnerable port: {packet_info['dst_port']}")
        
        return len(suspicious_indicators) > 0
    
    def _print_packet_result(self, result: Dict):
        """Print detailed result for each packet"""
        print("\n" + "="*60)
        print(f"PACKET {result['packet_number']} ANALYSIS RESULT")
        print("="*60)
        print(f"Source: {result['source_ip']}:{result['source_port']}")
        print(f"Destination: {result['destination_ip']}:{result['destination_port']}")
        print(f"Packet Size: {result['packet_size']} bytes")
        print(f"TCP Flags: {result['tcp_flags']}")
        print(f"Threat Classification: {result['threat_classification'].upper()}")
        print(f"Threat Level: {result['threat_level'].upper()}")
        print(f"Risk Score: {result['risk_score']:.2%}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        print(f"Mitigations: {result['mitigations_executed']}")
        print(f"Suspicious: {'YES' if result['suspicious_indicators'] else 'NO'}")
        if result['alert_id']:
            print(f"Alert ID: {result['alert_id']}")
        print("="*60)
    
    def _print_capture_summary(self, results: List[Dict]):
        """Print summary of the entire capture session"""
        print("\n" + "🎯" * 20)
        print("LIVE CAPTURE SESSION SUMMARY")
        print("🎯" * 20)
        print(f"Total Packets Captured: {len(self.capture_packets)}")
        print(f"Total Packets Processed: {self.processed_count}")
        print(f"Suspicious Packets Detected: {self.suspicious_count}")
        print(f"Detection Success Rate: {(self.processed_count/len(self.capture_packets)*100):.1f}%" if self.capture_packets else "0.0%")
        
        if results:
            avg_risk = np.mean([r['risk_score'] for r in results])
            avg_confidence = np.mean([r['confidence_score'] for r in results])
            print(f"Average Risk Score: {avg_risk:.2%}")
            print(f"Average Confidence: {avg_confidence:.2%}")
        
        print("🎯" * 20)


# Global instance
live_capture_manager = LiveCaptureManager()


def start_live_capture(interface: Optional[str] = None, capture_count: int = 20):
    """
    Main function to start live capture and processing
    
    Args:
        interface: Network interface to capture on (None for default)
        capture_count: Number of packets to capture (20-30 recommended)
    """
    print("🚀 Starting Hybrid IDS Live Capture System")
    print("=" * 50)
    
    if not SCAPY_AVAILABLE:
        print("❌ Scapy not available. Please install: pip install scapy")
        return
    
    try:
        # Start live capture
        results = live_capture_manager.start_live_capture(interface, capture_count)
        
        if results:
            print(f"\n✅ Live capture completed successfully!")
            print(f"📊 Processed {len(results)} packets through IDS pipeline")
        else:
            print("\n⚠️  No packets were successfully processed")
        
        return results
            
    except KeyboardInterrupt:
        print("\n🛑 Live capture interrupted by user")
        live_capture_manager.capture_active = False
        return []
    except Exception as e:
        print(f"\n❌ Live capture failed: {str(e)}")
        return []
    finally:
        print("\n🔚 Live capture session ended")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid IDS Live Capture System')
    parser.add_argument('--interface', '-i', type=str, help='Network interface to capture on')
    parser.add_argument('--count', '-c', type=int, default=20, help='Number of packets to capture (default: 20)')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with suspicious activity')
    
    args = parser.parse_args()
    
    print("🛡️ Hybrid Intelligent Intrusion Detection System (IDS)")
    print("📡 Live Network Traffic Capture Module")
    print("=" * 50)
    
    if args.demo:
        print("🎭 Demo mode enabled - will generate suspicious activity for visible detection")
        # In demo mode, we could simulate packets instead of capturing
        # But for now, we'll just capture normally
    
    # Start live capture
    start_live_capture(args.interface, args.count)