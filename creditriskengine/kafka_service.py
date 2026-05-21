"""
Enterprise Event Streaming Service (Apache Kafka Simulation)
Developed by Inclusion Algorithm Team

This module simulates an Apache Kafka event streaming cluster.
In a production banking environment, components publish events (like loan approvals) 
to Kafka topics. Other microservices (like Audit, Compliance, or Data Lake ingestion) 
consume these events asynchronously.
"""

import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KafkaSim")

class KafkaBrokerMock:
    """Simulates a centralized Kafka message broker with topics."""
    def __init__(self):
        self.topics: Dict[str, List[Dict[str, Any]]] = {
            "credit-decisions": [],
            "system-alerts": [],
            "ifrs9-recalculations": []
        }

    def produce(self, topic: str, message: Dict[str, Any]):
        if topic not in self.topics:
            self.topics[topic] = []
        
        # Add metadata like Kafka would
        enriched_msg = {
            "offset": len(self.topics[topic]),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": message
        }
        self.topics[topic].append(enriched_msg)
        logger.info(f"[KAFKA] Produced to topic '{topic}': {json.dumps(message)[:100]}...")

    def consume(self, topic: str, max_messages: int = 100) -> List[Dict[str, Any]]:
        """Simulates consuming the latest messages from a topic."""
        if topic not in self.topics:
            return []
        # Return newest first
        return sorted(self.topics[topic], key=lambda x: x["timestamp"], reverse=True)[:max_messages]

# Singleton broker for the application lifecycle
broker = KafkaBrokerMock()

class AuditEventProducer:
    """Producer client to publish credit decisions to the audit topic."""
    
    @staticmethod
    def publish_decision(loan_data: dict, decision_result: dict, client_ip: str = "127.0.0.1"):
        event = {
            "event_type": "CREDIT_DECISION",
            "source_system": "CreditRiskEngine_API",
            "client_ip": client_ip,
            "decision": decision_result.get("recommendation", "UNKNOWN"),
            "probability_of_default": decision_result.get("probability_of_default"),
            "risk_score": decision_result.get("risk_score"),
            "loan_amount": loan_data.get("amount_usd"),
            "term": loan_data.get("term_months"),
            "product_code": loan_data.get("product_code")
        }
        broker.produce("credit-decisions", event)

class SystemAlertProducer:
    @staticmethod
    def publish_alert(level: str, message: str):
        event = {
            "event_type": "SYSTEM_ALERT",
            "level": level,
            "message": message
        }
        broker.produce("system-alerts", event)
