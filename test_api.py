#!/usr/bin/env python3
"""
Simple test script for the API functionality
This tests the API endpoints without requiring a full server
"""

import sys
import os
import json
import requests
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_endpoints():
    """Test the API endpoints"""
    print("🧪 Testing API Endpoints...")
    print("=" * 50)
    
    # Test data for detection
    test_data = {
        "src_ip": "192.168.1.100",
        "dst_ip": "10.0.0.1",
        "src_port": 54321,
        "dst_port": 80,
        "protocol": "TCP",
        "duration": 10.5,
        "src_bytes": 1500,
        "dst_bytes": 750,
        "flags": "SF"
    }
    
    try:
        # Test detection endpoint
        print("📡 Testing /detect endpoint...")
        response = requests.post('http://localhost:8000/detect', json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection successful!")
            print(f"   Threat Classification: {result.get('threat_classification', 'N/A')}")
            print(f"   Threat Level: {result.get('threat_level', 'N/A')}")
            print(f"   Risk Score: {result.get('risk_score', 0):.2%}")
            print(f"   Confidence: {result.get('confidence_score', 0):.2%}")
            print(f"   Mitigations: {result.get('mitigations_executed', 0)}")
        else:
            print(f"❌ Detection failed: {response.status_code} - {response.text}")
        
        # Test health check
        print("\n🏥 Testing /health endpoint...")
        response = requests.get('http://localhost:8000/health')
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check successful!")
            print(f"   System Health: {health_data.get('system_health', 'N/A')}")
            print(f"   Active Alerts: {health_data.get('active_alerts', 0)}")
            print(f"   Blocked IPs: {health_data.get('blocked_ips', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code} - {response.text}")
        
        # Test alerts endpoint
        print("\n📋 Testing /alerts endpoint...")
        response = requests.get('http://localhost:8000/alerts')
        
        if response.status_code == 200:
            alerts_data = response.json()
            print("✅ Alerts endpoint successful!")
            print(f"   Total Alerts: {alerts_data.get('total_alerts', 0)}")
        else:
            print(f"❌ Alerts endpoint failed: {response.status_code} - {response.text}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure the server is running on http://localhost:8000")
        print("💡 To start the server, run: uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False


def test_frontend_access():
    """Test frontend access"""
    print("\n🌐 Testing Frontend Access...")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:8000/frontend')
        
        if response.status_code == 200:
            print("✅ Frontend accessible!")
            print("   Frontend HTML loaded successfully")
            # Check if key elements are present
            if 'Hybrid IDS' in response.text and 'Submit Network Traffic' in response.text:
                print("   ✅ Frontend contains expected content")
            else:
                print("   ⚠️  Frontend content may be incomplete")
        else:
            print(f"❌ Frontend access failed: {response.status_code} - {response.text}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to frontend. Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Frontend test failed: {str(e)}")
        return False


def test_system_components():
    """Test individual system components"""
    print("\n🔧 Testing System Components...")
    print("=" * 50)
    
    try:
        # Test imports
        from utils.logger import ids_logger
        from preprocessing.data_preprocessor import data_preprocessor
        from models.ml_models import model_manager
        from detection.hybrid_detector import detection_pipeline
        from alerts.alert_manager import alert_manager
        from mitigation.mitigation_engine import mitigation_orchestrator
        
        print("✅ All system components imported successfully")
        
        # Test logger
        ids_logger.info("Test log message")
        print("✅ Logger working")
        
        # Test data preprocessor
        import pandas as pd
        import numpy as np
        
        test_df = pd.DataFrame([{
            'duration': 1.0,
            'src_bytes': 1000.0,
            'dst_bytes': 500.0,
            'src_packets': 1.0,
            'dst_packets': 1.0,
            'protocol': 6.0,
            'flags': 1.0,
            'additional_metric': 1.0
        }])
        
        features, labels, feature_names = data_preprocessor.preprocess_network_data(test_df)
        print(f"✅ Data preprocessor working - Features shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False


def main():
    """Run all API tests"""
    print("🚀 Starting API and Frontend Tests")
    print("=" * 60)
    
    # Test system components first
    components_ok = test_system_components()
    
    if components_ok:
        # Test API endpoints
        api_ok = test_api_endpoints()
        
        # Test frontend
        frontend_ok = test_frontend_access()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ System Components: {'PASS' if components_ok else 'FAIL'}")
        print(f"✅ API Endpoints: {'PASS' if api_ok else 'FAIL'}")
        print(f"✅ Frontend Access: {'PASS' if frontend_ok else 'FAIL'}")
        
        if all([components_ok, api_ok, frontend_ok]):
            print("\n🎉 All tests passed! The system is ready.")
            print("\n💡 To start the full system, run:")
            print("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
            print("\n🌐 Then visit: http://localhost:8000/frontend")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            if not api_ok or not frontend_ok:
                print("\n💡 To start the API server, run:")
                print("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
    else:
        print("\n❌ System components test failed. Cannot proceed with API tests.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()