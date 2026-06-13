import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import threading
import time
from enum import Enum
from dataclasses import dataclass
from utils.logger import ids_logger

# Import other modules
from alerts.alert_manager import Alert, AlertManager, AlertNotifier
from detection.hybrid_detector import DetectionResult


class MitigationAction(Enum):
    """Available mitigation actions"""
    LOG_EVENT = "log_event"
    MONITOR_TRAFFIC = "monitor_traffic"
    QUARANTINE_SESSION = "quarantine_session"
    BLOCK_IP = "block_ip"
    NOTIFY_SECURITY_TEAM = "notify_security_team"
    ISOLATE_SYSTEM = "isolate_system"


class MitigationStatus(Enum):
    """Mitigation action status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MitigationActionRecord:
    """Record of a mitigation action"""
    action_id: str
    action_type: MitigationAction
    status: MitigationStatus
    target_ip: str
    alert_id: str
    timestamp: datetime
    duration: Optional[float]
    result: Optional[str]
    error_message: Optional[str]


class MitigationEngine:
    """Automated response and mitigation engine"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Mitigation Engine initialized")
        
        # State management
        self.active_mitigations: Dict[str, MitigationActionRecord] = {}
        self.blocked_ips: set = set()
        self.quarantined_sessions: Dict[str, datetime] = {}
        self.system_state = {
            'blocked_ips_count': 0,
            'quarantined_sessions_count': 0,
            'total_mitigations': 0,
            'active_mitigations': 0
        }
        
        # Configuration
        self.mitigation_config = {
            'quarantine_duration_minutes': 30,
            'auto_block_threshold': 0.8,
            'max_concurrent_mitigations': 10,
            'enable_auto_mitigation': True
        }
        
        # Threading for background tasks
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
    
    def process_alert(self, alert: Alert) -> List[MitigationActionRecord]:
        """
        Process an alert and execute appropriate mitigation actions
        
        Args:
            alert: Alert to process
            
        Returns:
            List of executed mitigation actions
        """
        self.logger.info(f"Processing alert {alert.alert_id} for mitigation")
        
        executed_actions = []
        
        # Check if auto-mitigation is enabled
        if not self.mitigation_config['enable_auto_mitigation']:
            self.logger.info("Auto-mitigation disabled, skipping automated response")
            return executed_actions
        
        # Determine appropriate actions based on severity and threat type
        actions_to_execute = self._determine_mitigation_actions(alert)
        
        # Execute actions
        for action_type in actions_to_execute:
            action_record = self._execute_mitigation_action(action_type, alert)
            if action_record:
                executed_actions.append(action_record)
        
        # Update system state
        self._update_system_state()
        
        self.logger.info(f"Completed mitigation for alert {alert.alert_id}. Executed {len(executed_actions)} actions")
        return executed_actions
    
    def _determine_mitigation_actions(self, alert: Alert) -> List[MitigationAction]:
        actions = []

        # Always log + monitor
        actions.append(MitigationAction.LOG_EVENT)
        actions.append(MitigationAction.MONITOR_TRAFFIC)

        # 🔥 USE RISK SCORE INSTEAD OF SEVERITY
        risk = getattr(alert, "risk_score", 0)

        # HIGH RISK
        if risk > 0.7:
            actions.append(MitigationAction.BLOCK_IP)
            actions.append(MitigationAction.QUARANTINE_SESSION)
            actions.append(MitigationAction.NOTIFY_SECURITY_TEAM)

            if alert.threat_type in ['malicious', 'data_exfiltration']:
                actions.append(MitigationAction.ISOLATE_SYSTEM)

        # MEDIUM RISK
        elif risk > 0.5:
            actions.append(MitigationAction.NOTIFY_SECURITY_TEAM)

        # LOW → do nothing extra

        # Optional: threat-specific overrides
        if 'brute_force' in alert.threat_type.lower():
            actions.append(MitigationAction.BLOCK_IP)

        if 'dos' in alert.threat_type.lower() and risk > 0.6:
            actions.append(MitigationAction.BLOCK_IP)
            actions.append(MitigationAction.QUARANTINE_SESSION)

        return list(set(actions))
    
    def _execute_mitigation_action(self, action_type: MitigationAction, 
                                 alert: Alert) -> Optional[MitigationActionRecord]:
        """Execute a specific mitigation action"""
        action_id = f"{alert.alert_id}_{action_type.value}_{int(time.time())}"
        
        action_record = MitigationActionRecord(
            action_id=action_id,
            action_type=action_type,
            status=MitigationStatus.PENDING,
            target_ip=alert.source_ip,
            alert_id=alert.alert_id,
            timestamp=datetime.now(),
            duration=None,
            result=None,
            error_message=None
        )
        
        self.active_mitigations[action_id] = action_record
        
        try:
            start_time = time.time()
            action_record.status = MitigationStatus.IN_PROGRESS
            
            # Execute the action
            if action_type == MitigationAction.LOG_EVENT:
                result = self._log_event(alert)
            elif action_type == MitigationAction.MONITOR_TRAFFIC:
                result = self._monitor_traffic(alert)
            elif action_type == MitigationAction.QUARANTINE_SESSION:
                result = self._quarantine_session(alert)
            elif action_type == MitigationAction.BLOCK_IP:
                result = self._block_ip(alert.source_ip)
            elif action_type == MitigationAction.NOTIFY_SECURITY_TEAM:
                result = self._notify_security_team(alert)
            elif action_type == MitigationAction.ISOLATE_SYSTEM:
                result = self._isolate_system(alert)
            else:
                raise ValueError(f"Unknown mitigation action: {action_type}")
            
            # Update action record
            end_time = time.time()
            action_record.duration = end_time - start_time
            action_record.status = MitigationStatus.COMPLETED
            action_record.result = result
            
            self.logger.info(f"Successfully executed {action_type.value} for alert {alert.alert_id}")
            
        except Exception as e:
            action_record.status = MitigationStatus.FAILED
            action_record.error_message = str(e)
            self.logger.error(f"Failed to execute {action_type.value}: {str(e)}")
        
        return action_record
    
    def _log_event(self, alert: Alert) -> str:
        """Log security event"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'alert_id': alert.alert_id,
            'severity': alert.severity.value,
            'source_ip': alert.source_ip,
            'threat_type': alert.threat_type,
            'risk_score': alert.risk_score,
            'description': alert.description
        }
        
        # In a real implementation, this would write to a SIEM or log management system
        self.logger.warning(f"SECURITY EVENT LOGGED: {json.dumps(log_entry)}")
        return f"Event logged for alert {alert.alert_id}"
    
    def _monitor_traffic(self, alert: Alert) -> str:
        """Enhanced traffic monitoring for the source IP"""
        # In a real implementation, this would configure network monitoring tools
        self.logger.info(f"Enhanced monitoring enabled for {alert.source_ip}")
        return f"Traffic monitoring activated for {alert.source_ip}"
    
    def _quarantine_session(self, alert: Alert) -> str:
        """Quarantine the network session"""
        session_id = f"sess_{alert.source_ip}_{int(time.time())}"
        quarantine_duration = self.mitigation_config['quarantine_duration_minutes']
        
        self.quarantined_sessions[session_id] = datetime.now() + timedelta(minutes=quarantine_duration)
        
        # In a real implementation, this would interact with network devices/firewalls
        self.logger.warning(f"Session {session_id} quarantined for {quarantine_duration} minutes")
        return f"Session quarantined: {session_id}"
    
    def _block_ip(self, ip_address: str) -> str:
        """Block an IP address"""
        if ip_address in self.blocked_ips:
            return f"IP {ip_address} already blocked"
        
        self.blocked_ips.add(ip_address)
        
        # In a real implementation, this would configure firewall rules
        self.logger.critical(f"IP BLOCKED: {ip_address}")
        return f"IP {ip_address} blocked successfully"
    
    def _notify_security_team(self, alert: Alert) -> str:
        """Notify security team of critical alert"""
        # In a real implementation, this would send notifications to security personnel
        message = f"CRITICAL ALERT: {alert.threat_type} detected from {alert.source_ip} with risk score {alert.risk_score:.1%}"
        self.logger.critical(message)
        return f"Security team notified for alert {alert.alert_id}"
    
    def _isolate_system(self, alert: Alert) -> str:
        """Isolate affected system from network"""
        # In a real implementation, this would disconnect the system from the network
        self.logger.critical(f"SYSTEM ISOLATION: Isolating systems affected by {alert.source_ip}")
        return f"System isolation initiated for alert {alert.alert_id}"
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        self._update_system_state()
        return self.system_state.copy()
    
    def _update_system_state(self) -> None:
        """Update system state metrics"""
        current_time = datetime.now()
        
        # Clean up expired quarantines
        expired_sessions = [
            session_id for session_id, expiry_time in self.quarantined_sessions.items()
            if expiry_time < current_time
        ]
        for session_id in expired_sessions:
            del self.quarantined_sessions[session_id]
        
        # Update state
        self.system_state.update({
            'blocked_ips_count': len(self.blocked_ips),
            'quarantined_sessions_count': len(self.quarantined_sessions),
            'total_mitigations': len([m for m in self.active_mitigations.values() 
                                    if m.status == MitigationStatus.COMPLETED]),
            'active_mitigations': len([m for m in self.active_mitigations.values() 
                                     if m.status in [MitigationStatus.PENDING, MitigationStatus.IN_PROGRESS]])
        })
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if an IP address is blocked"""
        return ip_address in self.blocked_ips
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            self.logger.info(f"IP {ip_address} unblocked")
            return True
        return False
    
    def get_mitigation_history(self) -> List[MitigationActionRecord]:
        """Get history of all mitigation actions"""
        return list(self.active_mitigations.values())
    
    def get_blocked_ips(self) -> List[str]:
        """Get list of blocked IP addresses"""
        return list(self.blocked_ips)
    
    def get_quarantined_sessions(self) -> Dict[str, datetime]:
        """Get quarantined sessions with expiry times"""
        return self.quarantined_sessions.copy()
    
    def _maintenance_loop(self) -> None:
        """Background maintenance tasks"""
        while True:
            try:
                # Clean up old mitigation records (keep last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                old_records = [
                    action_id for action_id, record in self.active_mitigations.items()
                    if record.timestamp < cutoff_time
                ]
                
                for action_id in old_records:
                    del self.active_mitigations[action_id]
                
                # Update system state
                self._update_system_state()
                
                # Log system status every 10 minutes
                if int(time.time()) % 600 == 0:  # Every 10 minutes
                    self.logger.info(f"System state: {self.system_state}")
                
            except Exception as e:
                self.logger.error(f"Error in maintenance loop: {str(e)}")
            
            time.sleep(60)  # Run every minute
    
    def _format_action_details(self, mitigation_actions: List[str]) -> List[str]:
        """Format mitigation actions into human-readable list"""
        action_details = []
        
        for action in mitigation_actions:
            if "blocked" in action.lower():
                # Extract IP from block action
                import re
                ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', action)
                if ip_match:
                    action_details.append(f"Blocked IP: {ip_match.group(1)}")
                else:
                    action_details.append(action)
            elif "quarantined" in action.lower():
                action_details.append(action)
            elif "notified" in action.lower() or "alert" in action.lower():
                action_details.append("Alert sent to admin")
            elif "logged" in action.lower():
                action_details.append("Traffic logged")
            elif "isolated" in action.lower():
                action_details.append("System isolated")
            else:
                action_details.append(action)
        
        return action_details
    
    def export_mitigation_report(self) -> Dict[str, Any]:
        """Export mitigation activity report"""
        history = self.get_mitigation_history()
        
        # Group by status
        status_counts = {}
        for status in MitigationStatus:
            status_counts[status.value] = len([m for m in history if m.status == status])
        
        # Group by action type
        action_counts = {}
        for action in MitigationAction:
            action_counts[action.value] = len([m for m in history if m.action_type == action])
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'system_state': self.get_system_state(),
            'mitigation_statistics': {
                'total_actions': len(history),
                'status_distribution': status_counts,
                'action_distribution': action_counts,
                'success_rate': len([m for m in history if m.status == MitigationStatus.COMPLETED]) / len(history) if history else 0
            },
            'current_blocks': {
                'blocked_ips': self.get_blocked_ips(),
                'quarantined_sessions': list(self.get_quarantined_sessions().keys())
            },
            'recent_actions': [
                {
                    'action_id': m.action_id,
                    'action_type': m.action_type.value,
                    'status': m.status.value,
                    'target_ip': m.target_ip,
                    'timestamp': m.timestamp.isoformat(),
                    'duration': m.duration
                }
                for m in sorted(history, key=lambda x: x.timestamp, reverse=True)[:50]
            ]
        }


class MitigationOrchestrator:
    """Orchestrates mitigation responses across the system"""
    
    def __init__(self):
        self.logger = ids_logger
        self.logger.info("Mitigation Orchestrator initialized")
        
        self.mitigation_engine = MitigationEngine()
        self.alert_manager = AlertManager()
        self.alert_notifier = AlertNotifier()
    
    def handle_detection_result(self, detection_result: DetectionResult,
                               source_ip: str,
                               destination_ip: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle a complete detection result with full mitigation workflow
        
        Args:
            detection_result: Detection result from hybrid detector
            source_ip: Source IP address
            destination_ip: Destination IP address
            
        Returns:
            Complete response with alert and mitigation details
        """
        self.logger.info(f"Handling detection result for {source_ip}")
        
        try:
            # Step 1: Create alert
            alert = self.alert_manager.create_alert_from_detection(
                detection_result, source_ip, destination_ip
            )
            
            # Step 2: Send notifications
            notifications = self.alert_notifier.notify_alert(alert)
            
            # Step 3: Execute mitigations
            mitigations = self.mitigation_engine.process_alert(alert)
            
            # Step 4: Generate response
            response = {
                'alert_created': True,
                'alert_id': alert.alert_id,
                'alert_summary': self.alert_manager.get_alert_summary(alert),
                'notifications_sent': notifications,
                'mitigations_executed': len(mitigations),
                'mitigation_details': [
                    {
                        'action_id': m.action_id,
                        'action_type': m.action_type.value,
                        'status': m.status.value,
                        'timestamp': m.timestamp.isoformat()
                    }
                    for m in mitigations
                ],
                'system_state': self.mitigation_engine.get_system_state()
            }
            
            self.logger.info(f"Complete response generated for {source_ip}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling detection result: {str(e)}")
            return {
                'alert_created': False,
                'error': str(e),
                'system_state': self.mitigation_engine.get_system_state()
            }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'alert_statistics': self.alert_manager.get_alert_statistics(),
            'system_state': self.mitigation_engine.get_system_state(),
            'blocked_ips': self.mitigation_engine.get_blocked_ips(),
            'quarantined_sessions': len(self.mitigation_engine.get_quarantined_sessions()),
            'recent_mitigations': self.mitigation_engine.export_mitigation_report()
        }


# Global instances
mitigation_engine = MitigationEngine()
mitigation_orchestrator = MitigationOrchestrator()