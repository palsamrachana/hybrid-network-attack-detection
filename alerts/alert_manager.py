import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import uuid
from enum import Enum
from utils.logger import ids_logger

# Import other modules
from detection.hybrid_detector import DetectionResult, ThreatLevel


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""
    NEW = "new"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    status: AlertStatus
    source_ip: str
    destination_ip: Optional[str]
    threat_type: str
    threat_level: ThreatLevel
    confidence_score: float
    risk_score: float
    description: str
    detection_method: str  # "ml", "rule_based", "hybrid"
    affected_assets: List[str]
    mitigation_actions: List[str]
    explanation: str
    raw_data: Dict[str, Any]


class AlertManager:
    """Manages alert creation, classification, and lifecycle"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Alert Manager initialized")
        
        # Alert storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert configuration
        self.severity_thresholds = {
            'low': (0.0, 0.3),
            'medium': (0.3, 0.6),
            'high': (0.6, 0.8),
            'critical': (0.8, 1.0)
        }
        
        # Mitigation actions mapping
        self.mitigation_mapping = {
            'low': ['log_event', 'monitor_traffic'],
            'medium': ['log_event', 'monitor_traffic', 'quarantine_session'],
            'high': ['log_event', 'monitor_traffic', 'quarantine_session', 'block_ip'],
            'critical': ['log_event', 'monitor_traffic', 'quarantine_session', 'block_ip', 'notify_security_team']
        }
    
    def create_alert_from_detection(self, detection_result: DetectionResult,
                                  source_ip: str, 
                                  destination_ip: Optional[str] = None,
                                  raw_data: Optional[Dict] = None) -> Alert:
        """
        Create an alert from a detection result
        
        Args:
            detection_result: DetectionResult from hybrid detector
            source_ip: Source IP address
            destination_ip: Destination IP address
            raw_data: Raw detection data
            
        Returns:
            Created Alert object
        """
        self.logger.info(f"Creating alert for {source_ip} - {detection_result.threat_classification}")
        
        # Determine severity
        severity = self._classify_severity(detection_result.risk_score)
        
        # Determine detection method
        detection_method = self._determine_detection_method(detection_result)
        
        # Generate mitigation actions
        mitigation_actions = self.mitigation_mapping.get(severity.value, [])
        
        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            severity=severity,
            status=AlertStatus.NEW,
            source_ip=source_ip,
            destination_ip=destination_ip,
            threat_type=detection_result.threat_classification,
            threat_level=detection_result.threat_level,
            confidence_score=detection_result.confidence_score,
            risk_score=detection_result.risk_score,
            description=self._generate_alert_description(detection_result, severity),
            detection_method=detection_method,
            affected_assets=self._identify_affected_assets(detection_result, source_ip, destination_ip),
            mitigation_actions=mitigation_actions,
            explanation=detection_result.explanation,
            raw_data=raw_data or {}
        )
        
        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert creation
        self.logger.warning(
            f"ALERT CREATED - ID: {alert.alert_id}, "
            f"Severity: {severity.value}, "
            f"Source: {source_ip}, "
            f"Threat: {detection_result.threat_classification}, "
            f"Risk: {detection_result.risk_score:.2%}"
        )
        
        return alert
    
    def _classify_severity(self, risk_score: float) -> AlertSeverity:
        """Classify alert severity based on risk score"""
        for severity_name, (min_val, max_val) in self.severity_thresholds.items():
            if min_val <= risk_score < max_val:
                return AlertSeverity(severity_name)
        
        # Default to high if no match
        return AlertSeverity.HIGH
    
    def _determine_detection_method(self, detection_result: DetectionResult) -> str:
        """Determine the primary detection method"""
        ml_confidence = detection_result.ml_confidence
        rule_confidence = len(detection_result.rule_based_alerts) * 0.1  # Simple weighting
        
        if ml_confidence > 0.7 and rule_confidence < 0.3:
            return "ml"
        elif rule_confidence > 0.5 and ml_confidence < 0.5:
            return "rule_based"
        else:
            return "hybrid"
    
    def _generate_alert_description(self, detection_result: DetectionResult, 
                                  severity: AlertSeverity) -> str:
        """Generate human-readable alert description"""
        base_description = f"Threat detected: {detection_result.threat_classification}"
        
        if detection_result.rule_based_alerts:
            alert_types = list(set([
                alert.get('attack_type', alert.get('anomaly_type', 'unknown'))
                for alert in detection_result.rule_based_alerts
            ]))
            base_description += f" with {len(detection_result.rule_based_alerts)} rule-based alerts: {', '.join(alert_types[:3])}"
        
        confidence_text = f"Confidence: {detection_result.confidence_score:.1%}"
        risk_text = f"Risk Score: {detection_result.risk_score:.1%}"
        
        return f"{base_description}. {confidence_text}. {risk_text}. Severity: {severity.value.upper()}"
    
    def _identify_affected_assets(self, detection_result: DetectionResult,
                                source_ip: str, destination_ip: Optional[str]) -> List[str]:
        """Identify potentially affected assets"""
        assets = []
        
        # Add IP addresses
        assets.append(f"IP: {source_ip}")
        if destination_ip:
            assets.append(f"IP: {destination_ip}")
        
        # Add based on threat type
        if detection_result.threat_classification in ['malicious', 'suspicious']:
            assets.extend([
                "Network Infrastructure",
                "Application Services",
                "Data Systems"
            ])
        
        # Add specific assets based on detected patterns
        for alert in detection_result.rule_based_alerts:
            if alert.get('subtype') == 'ssh_brute_force':
                assets.append("SSH Services")
            elif alert.get('subtype') == 'http_brute_force':
                assets.append("Web Services")
            elif alert.get('subtype') == 'data_exfiltration':
                assets.append("Data Storage Systems")
        
        return list(set(assets))  # Remove duplicates
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        return self.alerts.get(alert_id)
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get all alerts of a specific severity"""
        return [alert for alert in self.alerts.values() if alert.severity == severity]
    
    def get_alerts_by_status(self, status: AlertStatus) -> List[Alert]:
        """Get all alerts with a specific status"""
        return [alert for alert in self.alerts.values() if alert.status == status]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts.values() 
                if alert.timestamp > cutoff_time]
    
    def update_alert_status(self, alert_id: str, status: AlertStatus) -> bool:
        """Update alert status"""
        if alert_id in self.alerts:
            old_status = self.alerts[alert_id].status
            self.alerts[alert_id].status = status
            
            self.logger.info(
                f"Alert {alert_id} status updated: {old_status.value} -> {status.value}"
            )
            return True
        return False
    
    def add_mitigation_action(self, alert_id: str, action: str) -> bool:
        """Add a mitigation action to an alert"""
        if alert_id in self.alerts:
            if action not in self.alerts[alert_id].mitigation_actions:
                self.alerts[alert_id].mitigation_actions.append(action)
                self.logger.info(f"Added mitigation action '{action}' to alert {alert_id}")
                return True
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alerts)
        
        # Severity distribution
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(self.get_alerts_by_severity(severity))
        
        # Status distribution
        status_counts = {}
        for status in AlertStatus:
            status_counts[status.value] = len(self.get_alerts_by_status(status))
        
        # Recent activity (last 24 hours)
        recent_alerts = self.get_recent_alerts(24)
        recent_by_severity = {}
        for severity in AlertSeverity:
            recent_by_severity[severity.value] = len([
                alert for alert in recent_alerts if alert.severity == severity
            ])
        
        return {
            'total_alerts': total_alerts,
            'severity_distribution': severity_counts,
            'status_distribution': status_counts,
            'recent_alerts_24h': len(recent_alerts),
            'recent_by_severity': recent_by_severity,
            'average_risk_score': np.mean([alert.risk_score for alert in self.alerts.values()]) if self.alerts else 0.0,
            'highest_risk_alert': max(self.alerts.values(), key=lambda x: x.risk_score).alert_id if self.alerts else None
        }
    
    def export_alerts_to_json(self, filename: Optional[str] = None) -> str:
        """Export alerts to JSON format"""
        if filename is None:
            filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert alerts to serializable format
        alerts_data = []
        for alert in self.alerts.values():
            alert_dict = asdict(alert)
            # Convert datetime to string
            alert_dict['timestamp'] = alert.timestamp.isoformat()
            alert_dict['severity'] = alert.severity.value
            alert_dict['status'] = alert.status.value
            alert_dict['threat_level'] = alert.threat_level.value
            alerts_data.append(alert_dict)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_alerts': len(alerts_data),
                'alerts': alerts_data
            }, f, indent=2)
        
        self.logger.info(f"Exported {len(alerts_data)} alerts to {filename}")
        return filename
    
    def get_alert_summary(self, alert: Alert) -> Dict[str, Any]:
        """Get a summary of an alert"""
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
            'affected_assets_count': len(alert.affected_assets),
            'mitigation_actions_count': len(alert.mitigation_actions),
            'explanation_preview': alert.explanation[:200] + "..." if len(alert.explanation) > 200 else alert.explanation
        }
    
    def cleanup_old_alerts(self, days: int = 30) -> int:
        """Remove alerts older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        old_alerts = [alert_id for alert_id, alert in self.alerts.items() 
                     if alert.timestamp < cutoff_time]
        
        for alert_id in old_alerts:
            del self.alerts[alert_id]
        
        self.logger.info(f"Cleaned up {len(old_alerts)} alerts older than {days} days")
        return len(old_alerts)


class AlertNotifier:
    """Handles alert notifications and escalation"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Alert Notifier initialized")
        
        # Notification configuration
        self.notification_rules = {
            'high': ['email', 'sms', 'dashboard'],
            'critical': ['email', 'sms', 'dashboard', 'webhook']
        }
    
    def notify_alert(self, alert: Alert) -> Dict[str, bool]:
        """Send notifications for an alert"""
        self.logger.info(f"Sending notifications for alert {alert.alert_id}")
        
        notifications_sent = {}
        
        # Determine notification methods based on severity
        notification_methods = self.notification_rules.get(alert.severity.value, ['dashboard'])
        
        for method in notification_methods:
            try:
                success = self._send_notification(alert, method)
                notifications_sent[method] = success
            except Exception as e:
                self.logger.error(f"Failed to send {method} notification: {str(e)}")
                notifications_sent[method] = False
        
        return notifications_sent
    
    def _send_notification(self, alert: Alert, method: str) -> bool:
        """Send a specific type of notification"""
        if method == 'email':
            return self._send_email_notification(alert)
        elif method == 'sms':
            return self._send_sms_notification(alert)
        elif method == 'dashboard':
            return self._send_dashboard_notification(alert)
        elif method == 'webhook':
            return self._send_webhook_notification(alert)
        else:
            self.logger.warning(f"Unknown notification method: {method}")
            return False
    
    def _send_email_notification(self, alert: Alert) -> bool:
        """Send email notification (placeholder implementation)"""
        # In a real implementation, this would use an email service
        self.logger.info(f"EMAIL NOTIFICATION: Alert {alert.alert_id} - {alert.severity.value} severity")
        return True
    
    def _send_sms_notification(self, alert: Alert) -> bool:
        """Send SMS notification (placeholder implementation)"""
        # In a real implementation, this would use an SMS service
        self.logger.info(f"SMS NOTIFICATION: Alert {alert.alert_id} - {alert.severity.value} severity")
        return True
    
    def _send_dashboard_notification(self, alert: Alert) -> bool:
        """Send dashboard notification (placeholder implementation)"""
        # In a real implementation, this would update a dashboard system
        self.logger.info(f"DASHBOARD NOTIFICATION: Alert {alert.alert_id}")
        return True
    
    def _send_webhook_notification(self, alert: Alert) -> bool:
        """Send webhook notification (placeholder implementation)"""
        # In a real implementation, this would send HTTP POST to configured webhooks
        self.logger.info(f"WEBHOOK NOTIFICATION: Alert {alert.alert_id}")
        return True


# Global instances
alert_manager = AlertManager()
alert_notifier = AlertNotifier()