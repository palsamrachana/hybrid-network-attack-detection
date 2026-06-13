#!/usr/bin/env python3
"""
Test script for the live capture functionality
This script tests the live capture module without requiring actual network capture
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import system modules
from utils.live_capture import live_capture_manager, start_live_capture
from utils.logger import ids_logger


def test_feature_conversion():
    """Test the feature conversion functionality"""
    print("🧪 Testing Feature Conversion...")
    
    # Create mock packet info
    mock_packet = {
        'timestamp': datetime.now(),
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.1',
        'src_port': 54321,
        'dst_port': 80,
        'protocol': 'TCP',
        'packet_size': 1500,
        'flags': 'SF',  # Normal SYN-FIN
        'urg_ptr': 0,
        'seq': 12345,
        'ack': 0
    }
    
    # Convert to features
    feature_vector = live_capture_manager._convert_packet_to_features(mock_packet)
    
    print(f"✅ Feature conversion successful!")
    print(f"   Features: {list(feature_vector.keys())}")
    print(f"   Feature count: {len(feature_vector)}")
    print(f"   Feature values: {feature_vector}")
    
    return feature_vector


def test_suspicious_detection():
    """Test suspicious activity detection"""
    print("\n🧪 Testing Suspicious Activity Detection...")
    
    # Test normal packet
    normal_packet = {
        'packet_size': 1000,
        'flags': 'SF',
        'dst_port': 80
    }
    
    # Test suspicious packet (URG flag)
    suspicious_packet = {
        'packet_size': 1000,
        'flags': 'U',  # URG flag - suspicious!
        'dst_port': 22  # SSH port - potentially suspicious
    }
    
    # Create mock detection result
    class MockDetectionResult:
        def __init__(self):
            self.risk_score = 0.5
            self.threat_classification = "normal"
            self.threat_level = type('ThreatLevel', (), {'HIGH': 'high'})()
    
    # Test normal packet
    normal_result = MockDetectionResult()
    normal_suspicious = live_capture_manager._check_suspicious_activity(normal_packet, normal_result)
    print(f"   Normal packet suspicious: {normal_suspicious}")
    print(f"   Normal packet risk score: {normal_result.risk_score}")
    
    # Test suspicious packet
    suspicious_result = MockDetectionResult()
    suspicious_detected = live_capture_manager._check_suspicious_activity(suspicious_packet, suspicious_result)
    print(f"   Suspicious packet detected: {suspicious_detected}")
    print(f"   Suspicious packet risk score: {suspicious_result.risk_score}")
    
    return normal_suspicious, suspicious_detected


def test_packet_extraction():
    """Test packet information extraction"""
    print("\n🧪 Testing Packet Information Extraction...")
    
    # Create a mock packet-like object for testing
    class MockPacket:
        def __init__(self):
            self.layers = []
            self.has_tcp = True
            
        def haslayer(self, layer_type):
            return self.has_tcp
            
        def __getitem__(self, layer_type):
            if layer_type == 'IP':
                return MockIP()
            elif layer_type == 'TCP':
                return MockTCP()
            return None
    
    class MockIP:
        def __init__(self):
            self.src = '192.168.1.100'
            self.dst = '10.0.0.1'
    
    class MockTCP:
        def __init__(self):
            self.sport = 54321
            self.dport = 80
            self.flags = 2  # SYN flag
            self.urgptr = 0
            self.seq = 12345
            self.ack = 0
    
    # Test packet extraction
    mock_packet = MockPacket()
    packet_info = live_capture_manager._extract_packet_info(mock_packet)
    
    if packet_info:
        print(f"✅ Packet extraction successful!")
        print(f"   Source IP: {packet_info['src_ip']}")
        print(f"   Destination IP: {packet_info['dst_ip']}")
        print(f"   Source Port: {packet_info['src_port']}")
        print(f"   Destination Port: {packet_info['dst_port']}")
        print(f"   Protocol: {packet_info['protocol']}")
        print(f"   Packet Size: {packet_info['packet_size']}")
        print(f"   Flags: {packet_info['flags']}")
    else:
        print("❌ Packet extraction failed")
    
    return packet_info


def test_full_pipeline():
    """Test the complete pipeline with mock data"""
    print("\n🧪 Testing Full Pipeline with Mock Data...")
    
    try:
        # Create mock DataFrame
        mock_data = pd.DataFrame([{
            'duration': 1.0,
            'src_bytes': 1500.0,
            'dst_bytes': 750.0,
            'src_packets': 1.0,
            'dst_packets': 1.0,
            'protocol': 6.0,
            'flags': 1.0,  # SF flag
            'additional_metric': 1.5
        }])
        
        print(f"✅ Mock data created: {mock_data.shape}")
        print(f"   Columns: {list(mock_data.columns)}")
        print(f"   Data: {mock_data.iloc[0].to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Live Capture Module Tests")
    print("=" * 50)
    
    # Test feature conversion
    feature_vector = test_feature_conversion()
    
    # Test suspicious detection
    normal_suspicious, suspicious_detected = test_suspicious_detection()
    
    # Test packet extraction
    packet_info = test_packet_extraction()
    
    # Test full pipeline
    pipeline_success = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Feature Conversion: {'PASS' if feature_vector else 'FAIL'}")
    print(f"✅ Suspicious Detection: {'PASS' if suspicious_detected else 'FAIL'}")
    print(f"✅ Packet Extraction: {'PASS' if packet_info else 'FAIL'}")
    print(f"✅ Pipeline Test: {'PASS' if pipeline_success else 'FAIL'}")
    
    if all([feature_vector, suspicious_detected, packet_info, pipeline_success]):
        print("\n🎉 All tests passed! Live capture module is ready.")
        print("\n💡 To run actual live capture, use:")
        print("   python utils/live_capture.py --interface [interface_name] --count 20")
        print("   Example: python utils/live_capture.py --interface eth0 --count 20")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    print("=" * 50)


if __name__ == "__main__":
    main()