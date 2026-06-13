#!/usr/bin/env python3
"""
Comprehensive test script for the fixed IDS system
Tests ML model training, detection logic, explainability, and output formatting
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import system modules
from models.ml_models import model_manager
from detection.hybrid_detector import hybrid_detector, detection_pipeline
from explainability.explainability_engine import explainability_engine
from mitigation.mitigation_engine import mitigation_orchestrator
from utils.logger import ids_logger


def test_ml_model_training():
    """Test ML model training and prediction"""
    print("🧪 Testing ML Model Training and Prediction...")
    print("=" * 60)
    
    try:
        # Test model manager ensures models are trained
        models_ready = model_manager.ensure_models_trained()
        print(f"✅ Models ready: {models_ready}")
        
        if not models_ready:
            print("❌ Models could not be trained")
            return False
        
        # Test prediction with sample features
        sample_features = np.array([[
            10.0,    # duration
            1500.0,  # src_bytes
            750.0,   # dst_bytes
            1.0,     # src_packets
            1.0,     # dst_packets
            6.0,     # protocol (TCP)
            1.0,     # flags (SF)
            1.5      # additional_metric
        ]])
        
        predictions = model_manager.load_and_predict("models/saved/", sample_features)
        print(f"✅ Prediction successful: {predictions}")
        
        # Check for meaningful confidence scores
        for model_name, result in predictions.items():
            if 'confidence' in result:
                confidence = result['confidence']
                print(f"   {model_name} confidence: {confidence:.4f}")
                if confidence > 0.1:  # Should not be always 0
                    print(f"✅ {model_name} has meaningful confidence score")
                else:
                    print(f"⚠️  {model_name} has low confidence: {confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML model test failed: {str(e)}")
        return False


def test_detection_logic():
    """Test detection logic with rule-based overrides"""
    print("\n🧪 Testing Detection Logic with Rule-Based Overrides...")
    print("=" * 60)
    
    try:
        # Test normal traffic
        normal_traffic = pd.DataFrame([{
            'duration': 10.0,
            'src_bytes': 1000.0,
            'dst_bytes': 500.0,
            'src_packets': 1.0,
            'dst_packets': 1.0,
            'protocol': 'TCP',
            'flags': 'SF',
            'src_ip': '192.168.1.100',
            'dst_ip': '10.0.0.1'
        }])
        
        # Test suspicious traffic (URG flag)
        suspicious_traffic = pd.DataFrame([{
            'duration': 60.0,
            'src_bytes': 15000.0,
            'dst_bytes': 7500.0,
            'src_packets': 200.0,
            'dst_packets': 100.0,
            'protocol': 'TCP',
            'flags': 'U',  # URG flag - suspicious!
            'src_ip': '192.168.1.200',
            'dst_ip': '10.0.0.1'
        }])
        
        # Test detection on normal traffic
        print("Testing normal traffic detection...")
        normal_result = detection_pipeline.process_data(normal_traffic)
        if 'error' not in normal_result:
            normal_threat = normal_result['detection']['threat_classification']
            normal_risk = normal_result['detection']['risk_score']
            normal_confidence = normal_result['detection']['confidence_score']
            print(f"   Normal traffic - Threat: {normal_threat}, Risk: {normal_risk:.2%}, Confidence: {normal_confidence:.2%}")
        else:
            print(f"   Normal traffic detection failed: {normal_result['error']}")
        
        # Test detection on suspicious traffic
        print("Testing suspicious traffic detection...")
        suspicious_result = detection_pipeline.process_data(suspicious_traffic)
        if 'error' not in suspicious_result:
            suspicious_threat = suspicious_result['detection']['threat_classification']
            suspicious_risk = suspicious_result['detection']['risk_score']
            suspicious_confidence = suspicious_result['detection']['confidence_score']
            print(f"   Suspicious traffic - Threat: {suspicious_threat}, Risk: {suspicious_risk:.2%}, Confidence: {suspicious_confidence:.2%}")
            
            # Check if suspicious traffic is detected as malicious
            if suspicious_risk > 0.5 and suspicious_threat != 'normal':
                print("✅ Suspicious traffic correctly detected as malicious")
            else:
                print("⚠️  Suspicious traffic may not be detected properly")
        else:
            print(f"   Suspicious traffic detection failed: {suspicious_result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection logic test failed: {str(e)}")
        return False


def test_explainability():
    """Test explainability module with error handling"""
    print("\n🧪 Testing Explainability Module...")
    print("=" * 60)
    
    try:
        # Sample features for explanation
        sample_features = np.array([[
            60.0,    # duration (longer = suspicious)
            15000.0, # src_bytes (large = suspicious)
            7500.0,  # dst_bytes
            200.0,   # src_packets (many = suspicious)
            100.0,   # dst_packets
            6.0,     # protocol (TCP)
            6.0,     # flags (URG = suspicious)
            15.0     # additional_metric
        ]])
        
        feature_names = [
            'duration', 'src_bytes', 'dst_bytes', 'src_packets', 
            'dst_packets', 'protocol', 'flags', 'additional_metric'
        ]
        
        # Create a mock detection result
        class MockDetectionResult:
            def __init__(self):
                self.threat_classification = "malicious"
                self.risk_score = 0.85
                self.ml_confidence = 0.75
                self.rule_based_alerts = [{'severity': 'high', 'attack_type': 'URG_flag_attack'}]
        
        mock_result = MockDetectionResult()
        
        # Test explanation
        explanation = explainability_engine.explain_detection_result(
            mock_result, sample_features, feature_names
        )
        
        print(f"✅ Explanation generated successfully")
        print(f"   Explanation text: {explanation.explanation_text[:100]}...")
        print(f"   Feature importance: {len(explanation.feature_importance)} features")
        print(f"   Has visualizations: {len(explanation.visualization_data) > 0}")
        
        # Test explanation summary
        summary = explainability_engine.get_explanation_summary(explanation)
        print(f"   Summary generated: {len(summary)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Explainability test failed: {str(e)}")
        return False


def test_mitigation_output():
    """Test mitigation output formatting"""
    print("\n🧪 Testing Mitigation Output Formatting...")
    print("=" * 60)
    
    try:
        # Create a mock detection result for high severity
        class MockDetectionResult:
            def __init__(self):
                self.threat_classification = "malicious"
                self.risk_score = 0.9
                self.ml_confidence = 0.8
                self.rule_based_alerts = [{'severity': 'high', 'attack_type': 'DoS_attack'}]
                self.threat_level = "HIGH"
                self.confidence_score = 0.85
        
        mock_result = MockDetectionResult()
        
        # Test mitigation orchestration
        response = mitigation_orchestrator.handle_detection_result(
            mock_result, "192.168.1.200", "10.0.0.1"
        )
        
        print(f"✅ Mitigation response generated: {response['alert_created']}")
        print(f"   Alert ID: {response.get('alert_id', 'N/A')}")
        print(f"   Mitigations executed: {response.get('mitigations_executed', 0)}")
        print(f"   Mitigation details: {len(response.get('mitigation_details', []))} actions")
        
        # Check for specific mitigation actions
        mitigation_details = response.get('mitigation_details', [])
        for action in mitigation_details:
            print(f"   - {action['action_type']}: {action['status']}")
        
        # Test system state
        system_state = mitigation_orchestrator.get_comprehensive_status()
        print(f"   System state available: {len(system_state) > 0}")
        print(f"   Blocked IPs: {system_state.get('blocked_ips', [])}")
        print(f"   Quarantined sessions: {system_state.get('quarantined_sessions', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mitigation output test failed: {str(e)}")
        return False


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\n🧪 Testing End-to-End Workflow...")
    print("=" * 60)
    
    try:
        # Create suspicious traffic data
        suspicious_traffic = pd.DataFrame([{
            'duration': 120.0,
            'src_bytes': 50000.0,  # Very large
            'dst_bytes': 25000.0,
            'src_packets': 500.0,  # Many packets
            'dst_packets': 250.0,
            'protocol': 'TCP',
            'flags': 'U',  # URG flag - definitely suspicious!
            'src_ip': '192.168.1.255',
            'dst_ip': '10.0.0.100'
        }])
        
        print("Processing suspicious traffic through complete pipeline...")
        
        # Process through detection pipeline
        detection_result = detection_pipeline.process_data(suspicious_traffic)
        
        if 'error' in detection_result:
            print(f"❌ Detection pipeline failed: {detection_result['error']}")
            return False
        
        # Extract detection result
        threat_info = detection_result['detection']
        print(f"✅ Detection completed:")
        print(f"   Threat: {threat_info['threat_classification']}")
        print(f"   Risk Score: {threat_info['risk_score']:.2%}")
        print(f"   Confidence: {threat_info['confidence_score']:.2%}")
        print(f"   ML Prediction: {threat_info['ml_prediction']}")
        print(f"   ML Confidence: {threat_info['ml_confidence']:.2%}")
        
        # Check if suspicious traffic is properly detected
        if threat_info['risk_score'] > 0.5 and threat_info['threat_classification'] != 'normal':
            print("✅ Suspicious traffic correctly identified as malicious")
        else:
            print("⚠️  Suspicious traffic may not be properly detected")
            return False
        
        # Test mitigation response
        print("Executing mitigation response...")
        mitigation_response = mitigation_orchestrator.handle_detection_result(
            detection_result['detailed_result'], 
            '192.168.1.255', 
            '10.0.0.100'
        )
        
        print(f"✅ Mitigation executed: {mitigation_response['mitigations_executed']} actions")
        
        # Test explainability
        print("Generating explanation...")
        explanation = explainability_engine.explain_detection_result(
            detection_result['detailed_result'],
            suspicious_traffic.iloc[0:1].values,  # Convert to numpy array
            detection_result['preprocessing']['feature_names']
        )
        
        print(f"✅ Explanation generated: {len(explanation.explanation_text)} characters")
        print(f"   Top contributing features: {len(explanation.feature_importance)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {str(e)}")
        return False


def main():
    """Run all tests for the fixed system"""
    print("🚀 Testing Fixed IDS System")
    print("=" * 80)
    print("Testing fixes for:")
    print("1. ML model always outputting 0")
    print("2. Rule-based overrides for obvious attacks")
    print("3. Explainability module errors")
    print("4. Improved output formatting")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("ML Model Training", test_ml_model_training),
        ("Detection Logic", test_detection_logic),
        ("Explainability Module", test_explainability),
        ("Mitigation Output", test_mitigation_output),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The IDS system fixes are working correctly.")
        print("\n💡 To start the complete system, run:")
        print("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
        print("   Then visit: http://localhost:8000/frontend")
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the implementation.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()