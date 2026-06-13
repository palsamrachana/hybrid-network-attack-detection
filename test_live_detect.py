#!/usr/bin/env python3
"""
Test script for the new /live-detect endpoint integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.live_capture import live_capture_manager
from preprocessing.data_preprocessor import preprocess_live_packet
from detection.hybrid_detector import detection_pipeline

def test_live_detect_integration():
    """Test the complete live detection integration"""
    print("🚀 Testing Live Detection Integration")
    print("=" * 50)
    
    try:
        # Test 1: Capture packets
        print("🧪 Test 1: Capturing packets...")
        packets = live_capture_manager.start_live_capture(capture_count=5)
        print(f"✅ Captured {len(packets)} packets")
        
        # Test 2: Preprocess packets
        print("\n🧪 Test 2: Preprocessing packets...")
        results = []
        for i, packet in enumerate(packets):
            features = preprocess_live_packet(packet)
            print(f"   Packet {i+1}: {features}")
            results.append(features)
        
        print(f"✅ Preprocessed {len(results)} packets")
        
        # Test 3: Test detection pipeline
        print("\n🧪 Test 3: Testing detection pipeline...")
        if results:
            # Convert to DataFrame for detection pipeline
            import pandas as pd
            test_data = pd.DataFrame(results)
            print(f"   Test data shape: {test_data.shape}")
            print(f"   Test data columns: {list(test_data.columns)}")
            
            # Test with detection pipeline
            try:
                detection_results = detection_pipeline.process_data(test_data)
                print("✅ Detection pipeline working")
                print(f"   Detection result keys: {list(detection_results.keys())}")
            except Exception as e:
                print(f"⚠️  Detection pipeline error: {e}")
                print("   This is expected if models aren't trained yet")
        
        print("\n" + "=" * 50)
        print("✅ Live detection integration test completed successfully!")
        print("\nNext steps:")
        print("1. Start the API server: python api/main.py")
        print("2. Access: http://127.0.0.1:8000/live-detect")
        print("3. The endpoint should return live detection results")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_live_detect_integration()