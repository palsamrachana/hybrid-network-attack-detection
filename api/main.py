from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import uvicorn
import json
import logging

# Import system modules
from data.input_module import traffic_simulator, user_activity_logger
from preprocessing.data_preprocessor import data_preprocessor
from models.ml_models import model_manager
from detection.hybrid_detector import detection_pipeline
from explainability.explainability_engine import explainability_engine
from alerts.alert_manager import alert_manager, alert_notifier
from mitigation.mitigation_engine import mitigation_orchestrator
from utils.logger import ids_logger
from utils.live_capture import start_live_capture

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Hybrid Intelligent Intrusion Detection System (IDS)",
    description="A comprehensive IDS with ML, rule-based detection, and explainable AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class NetworkTrafficRequest(BaseModel):
    """Request model for network traffic analysis"""
    src_ip: str = Field(..., description="Source IP address")
    dst_ip: str = Field(..., description="Destination IP address")
    src_port: int = Field(..., description="Source port")
    dst_port: int = Field(..., description="Destination port")
    protocol: str = Field(..., description="Protocol (TCP/UDP/ICMP)")
    duration: float = Field(..., description="Connection duration")
    src_bytes: int = Field(..., description="Source bytes")
    dst_bytes: int = Field(..., description="Destination bytes")
    flags: str = Field(..., description="Connection flags")

class DetectionResponse(BaseModel):
    """Response model for detection results"""
    threat_classification: str
    risk_score: float
    confidence_score: float
    threat_level: str
    explanation: str
    alert_id: Optional[str]
    mitigations_executed: List[str]

class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    system_health: str
    active_alerts: int
    blocked_ips: int
    quarantined_sessions: int
    last_training: Optional[str]
    model_performance: Dict[str, float]

# Global state
system_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global system_initialized
    try:
        ids_logger.info("Starting Hybrid IDS System")
        
        # Initialize system components
        ids_logger.info("Initializing system components...")
        
        # Generate some sample data for training
        sample_data = traffic_simulator.generate_sample_data(1000)
        
        # Preprocess data
        features, labels, feature_names = data_preprocessor.create_feature_dataset(
            sample_data, 'label'
        )
        
        # Train models
        training_results = model_manager.train_and_save_models(
            features, labels, feature_names
        )
        
        ids_logger.info("System initialization completed successfully")
        system_initialized = True
        
    except Exception as e:
        ids_logger.error(f"System initialization failed: {str(e)}")
        system_initialized = False

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hybrid Intelligent Intrusion Detection System (IDS)",
        "version": "1.0.0",
        "status": "running" if system_initialized else "initialization_failed"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = mitigation_orchestrator.get_comprehensive_status()
        
        response = SystemStatusResponse(
            system_health="healthy" if system_initialized else "unhealthy",
            active_alerts=len([alert for alert in alert_manager.alerts.values() 
                             if alert.status == alert_manager.AlertStatus.NEW]),
            blocked_ips=len(mitigation_orchestrator.mitigation_engine.blocked_ips),
            quarantined_sessions=len(mitigation_orchestrator.mitigation_engine.quarantined_sessions),
            last_training=datetime.now().isoformat(),
            model_performance={}
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/detect", response_model=DetectionResponse)
async def detect_threat(request: NetworkTrafficRequest, background_tasks: BackgroundTasks):
    """
    Analyze network traffic and detect potential threats
    
    This endpoint processes network traffic data through the complete IDS pipeline:
    1. Data preprocessing and feature extraction
    2. Hybrid detection (ML + rule-based)
    3. Alert creation and management
    4. Automated mitigation
    5. Explainability analysis
    """
    try:
        # Convert request to DataFrame
        traffic_data = pd.DataFrame([{
            'src_ip': request.src_ip,
            'dst_ip': request.dst_ip,
            'src_port': request.src_port,
            'dst_port': request.dst_port,
            'protocol': request.protocol,
            'duration': request.duration,
            'src_bytes': request.src_bytes,
            'dst_bytes': request.dst_bytes,
            'flags': request.flags
        }])
        
        # Process through detection pipeline
        detection_results = detection_pipeline.process_data(traffic_data)
        
        if 'error' in detection_results:
            raise HTTPException(status_code=500, detail=detection_results['error'])
        
        # Extract detection result
        detection_result = detection_results['detailed_result']
        
        # Handle complete workflow with mitigation
        response_data = mitigation_orchestrator.handle_detection_result(
            detection_result, 
            request.src_ip, 
            request.dst_ip
        )
        
        if not response_data.get('alert_created', False):
            raise HTTPException(status_code=500, detail=response_data.get('error', 'Detection failed'))
        
        # Generate explainability analysis
        features = detection_results['preprocessing']['features_shape']
        feature_names = detection_results['preprocessing']['feature_names']
        
        explanation = explainability_engine.explain_detection_result(
            detection_result, 
            np.random.rand(1, len(feature_names)),  # Mock features for explanation
            feature_names
        )
        
        # Get detailed mitigation actions
        mitigation_actions = []
        if 'mitigation_details' in response_data:
            mitigation_actions = [
                m.get('result', f"{m.get('action_type', 'Action')} completed")
                for m in response_data['mitigation_details']
            ]
        
        # Format action details
        action_details = mitigation_orchestrator.mitigation_engine._format_action_details(mitigation_actions)
        
        # Return detection response
        return DetectionResponse(
            threat_classification=detection_result.threat_classification,
            risk_score=detection_result.risk_score,
            confidence_score=detection_result.confidence_score,
            threat_level=detection_result.threat_level.value,
            explanation=explanation.explanation_text,
            alert_id=response_data.get('alert_id'),
            mitigations_executed=action_details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ids_logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/alerts")
async def get_alerts():
    """Get all alerts"""
    try:
        alerts = []
        for alert in alert_manager.alerts.values():
            alerts.append({
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'status': alert.status.value,
                'source_ip': alert.source_ip,
                'threat_type': alert.threat_type,
                'risk_score': alert.risk_score,
                'description': alert.description
            })
        
        return {
            'total_alerts': len(alerts),
            'alerts': sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {str(e)}")

@app.get("/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get specific alert by ID"""
    try:
        alert = alert_manager.get_alert_by_id(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            'alert_id': alert.alert_id,
            'timestamp': alert.timestamp.isoformat(),
            'severity': alert.severity.value,
            'status': alert.status.value,
            'source_ip': alert.source_ip,
            'destination_ip': alert.destination_ip,
            'threat_type': alert.threat_type,
            'threat_level': alert.threat_level.value,
            'confidence_score': alert.confidence_score,
            'risk_score': alert.risk_score,
            'description': alert.description,
            'detection_method': alert.detection_method,
            'affected_assets': alert.affected_assets,
            'mitigation_actions': alert.mitigation_actions,
            'explanation': alert.explanation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alert: {str(e)}")

@app.put("/alerts/{alert_id}/status")
async def update_alert_status(alert_id: str, status: str):
    """Update alert status"""
    try:
        # Validate status
        valid_statuses = ['new', 'investigating', 'resolved', 'false_positive']
        if status.lower() not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        success = alert_manager.update_alert_status(alert_id, alert_manager.AlertStatus(status.lower()))
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"message": f"Alert {alert_id} status updated to {status}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update alert status: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        return mitigation_orchestrator.get_comprehensive_status()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.get("/system/blocked-ips")
async def get_blocked_ips():
    """Get list of blocked IP addresses"""
    try:
        blocked_ips = mitigation_orchestrator.mitigation_engine.get_blocked_ips()
        return {
            'blocked_ips_count': len(blocked_ips),
            'blocked_ips': blocked_ips
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blocked IPs: {str(e)}")

@app.post("/system/unblock-ip")
async def unblock_ip(ip: str):
    """Unblock an IP address"""
    try:
        success = mitigation_orchestrator.mitigation_engine.unblock_ip(ip)
        if success:
            return {"message": f"IP {ip} unblocked successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"IP {ip} not found in blocked list")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unblock IP: {str(e)}")

@app.get("/explain/{alert_id}")
async def get_explanation(alert_id: str):
    """Get explanation for a specific alert"""
    try:
        alert = alert_manager.get_alert_by_id(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Create mock features for explanation (in real system, would use actual features)
        mock_features = np.random.rand(1, 20)
        feature_names = [f"feature_{i}" for i in range(20)]
        
        # Get explanation
        explanation = explainability_engine.explain_detection_result(
            alert_manager.detection_pipeline.hybrid_detector.DetectionResult(
                threat_classification=alert.threat_type,
                risk_score=alert.risk_score,
                confidence_score=alert.confidence_score,
                ml_prediction="unknown",
                ml_confidence=alert.confidence_score,
                rule_based_alerts=[],
                combined_confidence=alert.confidence_score,
                threat_level=alert.threat_level,
                explanation=alert.explanation
            ),
            mock_features,
            feature_names
        )
        
        return {
            'alert_id': alert_id,
            'explanation_summary': explainability_engine.get_explanation_summary(explanation),
            'visualization_data': explanation.visualization_data,
            'explanation_text': explanation.explanation_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")

@app.get("/simulate")
async def simulate_detection(background_tasks: BackgroundTasks):
    """Simulate detection with sample data"""
    try:
        # Generate sample traffic
        sample_data = traffic_simulator.generate_sample_data(10)
        
        results = []
        for _, row in sample_data.iterrows():
            # Create request
            request = NetworkTrafficRequest(
                src_ip=row['src_ip'],
                dst_ip=row['dst_ip'],
                src_port=row['src_port'],
                dst_port=row['dst_port'],
                protocol=row['protocol'],
                duration=row['duration'],
                src_bytes=row['src_bytes'],
                dst_bytes=row['dst_bytes'],
                flags=row['flags']
            )
            
            # Process detection
            detection_response = await detect_threat(request, background_tasks)
            results.append({
                'source_ip': request.src_ip,
                'detection': detection_response
            })
        
        return {
            'simulation_results': results,
            'total_processed': len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/live-detect")
def live_detect():
    try:
        results = start_live_capture(capture_count=25)

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import FileResponse
@app.get("/frontend")
async def get_frontend():
    return FileResponse("frontend/index.html")
    
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )