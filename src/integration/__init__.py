"""
Integration layer for external systems (MQTT, REST API, Home Assistant).

This module provides comprehensive integration with Home Assistant and other
external systems for seamless occupancy prediction publishing.
"""

from .mqtt_publisher import MQTTPublisher, MQTTPublishResult, MQTTConnectionStatus
from .prediction_publisher import PredictionPublisher, PredictionPayload, SystemStatusPayload
from .discovery_publisher import DiscoveryPublisher, DeviceInfo, SensorConfig
from .mqtt_integration_manager import MQTTIntegrationManager, MQTTIntegrationStats

__all__ = [
    'MQTTPublisher',
    'MQTTPublishResult', 
    'MQTTConnectionStatus',
    'PredictionPublisher',
    'PredictionPayload',
    'SystemStatusPayload',
    'DiscoveryPublisher', 
    'DeviceInfo',
    'SensorConfig',
    'MQTTIntegrationManager',
    'MQTTIntegrationStats'
]