# WebSocket API Documentation

## Overview

The Home Assistant Occupancy Prediction System provides a comprehensive WebSocket API for real-time updates of predictions, system status, and alerts. The API features authentication, rate limiting, connection management, and automatic integration with the existing TrackingManager system.

## Features

- **Multiple WebSocket Endpoints**: `/ws/predictions`, `/ws/system-status`, `/ws/alerts`, `/ws/room/{room_id}`
- **API Key Authentication**: Secure token-based authentication
- **Rate Limiting**: Configurable per-connection and per-message limits
- **Connection Management**: Automatic cleanup, heartbeat, and reconnection support
- **Message Acknowledgments**: Reliable delivery for critical messages
- **Real-time Updates**: Automatic publishing of predictions, alerts, and system status
- **Room-specific Subscriptions**: Filter updates by specific rooms
- **Comprehensive Error Handling**: Detailed error messages and recovery mechanisms

## Quick Start

### 1. Server Configuration

The WebSocket API is automatically started by the TrackingManager when enabled in configuration:

```yaml
# config/config.yaml
tracking:
  websocket_api_enabled: true
  websocket_api_host: "0.0.0.0"
  websocket_api_port: 8766
  websocket_api_max_connections: 500
  websocket_api_max_messages_per_minute: 60
```

### 2. Basic Client Connection

```python
import asyncio
import json
import websockets

async def connect_to_api():
    uri = "ws://localhost:8766/ws/predictions"
    
    async with websockets.connect(uri) as websocket:
        # Authenticate
        auth_message = {
            "type": "authentication",
            "data": {
                "api_key": "your-api-key",
                "client_name": "MyClient",
                "capabilities": ["predictions", "alerts"],
                "room_filters": []  # Empty = access to all rooms
            }
        }
        await websocket.send(json.dumps(auth_message))
        
        # Subscribe to predictions
        subscribe_message = {
            "type": "subscribe",
            "data": {
                "endpoint": "/ws/predictions",
                "room_id": None,
                "filters": {}
            }
        }
        await websocket.send(json.dumps(subscribe_message))
        
        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['message_type']}")

asyncio.run(connect_to_api())
```

## WebSocket Endpoints

### 1. `/ws/predictions`
Real-time occupancy prediction updates for all rooms.

**Messages:**
- Prediction updates when new predictions are generated
- Model accuracy changes
- Prediction confidence updates

### 2. `/ws/system-status`
System health and performance monitoring.

**Messages:**
- System performance metrics
- Component health status
- Resource utilization updates

### 3. `/ws/alerts`
Critical system alerts and notifications.

**Messages:**
- Accuracy degradation alerts
- Model retraining notifications
- System error alerts
- Concept drift warnings

### 4. `/ws/room/{room_id}`
Room-specific prediction updates.

**Messages:**
- Predictions for the specific room only
- Room state changes
- Room-specific alerts

## Message Format

All WebSocket messages follow a standardized JSON format:

```json
{
  "message_id": "unique-uuid",
  "message_type": "prediction_update",
  "timestamp": "2024-01-15T14:30:00.000Z",
  "endpoint": "/ws/predictions",
  "room_id": "living_room",
  "requires_ack": false,
  "data": {
    // Message-specific data
  }
}
```

### Message Types

| Type | Description | Requires ACK |
|------|-------------|--------------|
| `connection` | Welcome message after connection | No |
| `authentication` | Authentication response | No |
| `subscription_status` | Subscription confirmation | No |
| `prediction_update` | New prediction available | No |
| `system_status_update` | System status change | No |
| `alert_notification` | Critical system alert | Yes |
| `drift_notification` | Concept drift detected | Yes |
| `heartbeat` | Server heartbeat | No |
| `error` | Error message | No |
| `rate_limit_warning` | Rate limit exceeded | No |

## Authentication

### API Key Authentication

All connections must authenticate using an API key:

```json
{
  "type": "authentication",
  "data": {
    "api_key": "your-api-key-here",
    "client_name": "YourClientName",
    "capabilities": ["predictions", "system_status", "alerts"],
    "room_filters": ["living_room", "bedroom"]
  }
}
```

**Response:**
```json
{
  "message_type": "authentication",
  "data": {
    "success": true,
    "message": "Authentication successful",
    "capabilities": ["predictions", "system_status", "alerts"],
    "room_filters": ["living_room", "bedroom"]
  }
}
```

### Capabilities

- `predictions`: Access to prediction updates
- `system_status`: Access to system status updates
- `alerts`: Access to alert notifications
- `room_specific`: Access to room-specific endpoints

### Room Filters

- Empty array `[]`: Access to all rooms
- Specific rooms `["living_room", "bedroom"]`: Access only to specified rooms

## Subscription Management

### Subscribe to Endpoint

```json
{
  "type": "subscribe",
  "data": {
    "endpoint": "/ws/predictions",
    "room_id": null,
    "filters": {}
  }
}
```

### Unsubscribe from Endpoint

```json
{
  "type": "unsubscribe",
  "data": {
    "endpoint": "/ws/predictions",
    "room_id": null
  }
}
```

## Message Examples

### Prediction Update

```json
{
  "message_id": "123e4567-e89b-12d3-a456-426614174000",
  "message_type": "prediction_update",
  "timestamp": "2024-01-15T14:30:00.000Z",
  "endpoint": "/ws/predictions",
  "room_id": "living_room",
  "requires_ack": false,
  "data": {
    "room_id": "living_room",
    "room_name": "Living Room",
    "predicted_time": "2024-01-15T15:45:00.000Z",
    "transition_type": "occupied",
    "confidence_score": 0.87,
    "time_until_seconds": 4500,
    "time_until_human": "1h 15m",
    "current_state": "vacant",
    "model_type": "ensemble",
    "model_version": "1.0.0",
    "alternatives": [
      {
        "predicted_time": "2024-01-15T15:30:00.000Z",
        "confidence": 0.71
      }
    ],
    "features_used": 25,
    "prediction_metadata": {
      "certainty_level": "high",
      "pattern_match": "weekday_evening"
    }
  }
}
```

### System Status Update

```json
{
  "message_id": "123e4567-e89b-12d3-a456-426614174001",
  "message_type": "system_status_update",
  "timestamp": "2024-01-15T14:30:00.000Z",
  "endpoint": "/ws/system-status",
  "requires_ack": false,
  "data": {
    "system_health": "healthy",
    "uptime_seconds": 86400,
    "active_connections": 15,
    "predictions_per_minute": 12,
    "model_accuracy": {
      "living_room": 0.89,
      "bedroom": 0.92,
      "kitchen": 0.84
    },
    "database_status": {
      "connected": true,
      "query_time_ms": 12.5
    },
    "mqtt_status": {
      "connected": true,
      "messages_published": 1450
    }
  }
}
```

### Alert Notification

```json
{
  "message_id": "123e4567-e89b-12d3-a456-426614174002",
  "message_type": "alert_notification",
  "timestamp": "2024-01-15T14:30:00.000Z",
  "endpoint": "/ws/alerts",
  "room_id": "bedroom",
  "requires_ack": true,
  "data": {
    "alert_type": "accuracy_degradation",
    "severity": "warning",
    "message": "Model accuracy has dropped below threshold",
    "room_id": "bedroom",
    "current_accuracy": 0.65,
    "threshold": 0.70,
    "recommended_action": "Model retraining recommended",
    "alert_id": "alert_123",
    "created_at": "2024-01-15T14:30:00.000Z"
  }
}
```

## Rate Limiting

The WebSocket API implements per-connection rate limiting:

- **Default Limit**: 60 messages per minute per connection
- **Rate Limit Warning**: Sent when approaching limit
- **Rate Limit Enforcement**: Temporary blocking when exceeded

### Rate Limit Warning

```json
{
  "message_type": "rate_limit_warning",
  "data": {
    "message": "Rate limit exceeded",
    "max_messages_per_minute": 60,
    "current_count": 58,
    "reset_time": "2024-01-15T14:31:00.000Z"
  }
}
```

## Connection Management

### Heartbeat

The server sends periodic heartbeats to maintain connection health:

```json
{
  "message_type": "heartbeat",
  "data": {
    "server_time": "2024-01-15T14:30:00.000Z",
    "connection_uptime": 3600.5
  }
}
```

Clients should respond with their own heartbeat:

```json
{
  "type": "heartbeat",
  "data": {
    "client_time": "2024-01-15T14:30:00.100Z"
  }
}
```

### Message Acknowledgment

Critical messages (alerts, drift notifications) require acknowledgment:

```json
{
  "type": "acknowledge",
  "data": {
    "message_id": "123e4567-e89b-12d3-a456-426614174002"
  }
}
```

## Error Handling

### Error Message Format

```json
{
  "message_type": "error",
  "data": {
    "error": "Authentication required for subscriptions",
    "error_code": "WEBSOCKET_AUTH_ERROR",
    "timestamp": "2024-01-15T14:30:00.000Z"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `WEBSOCKET_AUTH_ERROR` | Authentication failed |
| `WEBSOCKET_VALIDATION_ERROR` | Invalid message format |
| `WEBSOCKET_RATE_LIMIT_ERROR` | Rate limit exceeded |
| `WEBSOCKET_CONNECTION_ERROR` | Connection issue |
| `SUBSCRIPTION_ERROR` | Subscription failed |

## Configuration

### Server Configuration

```yaml
tracking:
  # WebSocket API settings
  websocket_api_enabled: true
  websocket_api_host: "0.0.0.0"
  websocket_api_port: 8766
  websocket_api_max_connections: 500
  websocket_api_max_messages_per_minute: 60
  websocket_api_heartbeat_interval_seconds: 30
  websocket_api_connection_timeout_seconds: 300
  websocket_api_message_acknowledgment_timeout_seconds: 30
```

### Environment Variables

```bash
# Override configuration via environment variables
WEBSOCKET_API_ENABLED=true
WEBSOCKET_API_HOST=0.0.0.0
WEBSOCKET_API_PORT=8766
WEBSOCKET_API_MAX_CONNECTIONS=500
```

## Client Libraries

### Python Client Example

See `examples/websocket_api_client.py` for a complete Python client implementation.

### JavaScript/Node.js Client

```javascript
const WebSocket = require('ws');

class OccupancyWebSocketClient {
  constructor(host, port, apiKey) {
    this.url = `ws://${host}:${port}/ws/predictions`;
    this.apiKey = apiKey;
    this.ws = null;
    this.authenticated = false;
  }

  async connect() {
    this.ws = new WebSocket(this.url);
    
    this.ws.on('open', () => {
      console.log('Connected to WebSocket API');
      this.authenticate();
    });

    this.ws.on('message', (data) => {
      const message = JSON.parse(data);
      this.handleMessage(message);
    });

    this.ws.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  authenticate() {
    const authMessage = {
      type: 'authentication',
      data: {
        api_key: this.apiKey,
        client_name: 'JSClient',
        capabilities: ['predictions', 'alerts'],
        room_filters: []
      }
    };
    
    this.ws.send(JSON.stringify(authMessage));
  }

  handleMessage(message) {
    switch (message.message_type) {
      case 'prediction_update':
        console.log(`Prediction for ${message.data.room_name}: ${message.data.transition_type} in ${message.data.time_until_human}`);
        break;
      case 'alert_notification':
        console.warn(`Alert: ${message.data.message}`);
        if (message.requires_ack) {
          this.acknowledgeMessage(message.message_id);
        }
        break;
      default:
        console.log(`Received ${message.message_type}`);
    }
  }

  acknowledgeMessage(messageId) {
    const ackMessage = {
      type: 'acknowledge',
      data: { message_id: messageId }
    };
    this.ws.send(JSON.stringify(ackMessage));
  }
}

// Usage
const client = new OccupancyWebSocketClient('localhost', 8766, 'your-api-key');
client.connect();
```

## Monitoring

### Connection Statistics

The WebSocket API provides comprehensive statistics via the TrackingManager:

```python
# Get WebSocket API status
status = tracking_manager.get_websocket_api_status()
print(f"Active connections: {status['active_connections']}")
print(f"Messages sent: {status['total_messages_sent']}")
print(f"Authentication failures: {status['authentication_failures']}")
```

### Metrics Available

- Active connections by endpoint
- Authentication success/failure rates
- Message throughput
- Rate limiting statistics
- Connection duration metrics
- Error rates by type

## Security Considerations

### API Key Management

- Store API keys securely
- Use environment variables for production
- Rotate keys regularly
- Monitor for unauthorized access

### Network Security

- Use WSS (WebSocket Secure) in production
- Implement proper firewall rules
- Consider VPN access for external clients
- Monitor connection patterns

### Rate Limiting

- Adjust limits based on client needs
- Monitor for abuse patterns
- Implement connection limits per IP
- Use exponential backoff for reconnections

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if WebSocket API is enabled in configuration
   - Verify host and port settings
   - Check firewall rules

2. **Authentication Failed**
   - Verify API key is correct
   - Check API key is enabled in configuration
   - Ensure client capabilities match server requirements

3. **Rate Limited**
   - Reduce message frequency
   - Implement proper rate limiting in client
   - Check server configuration limits

4. **Messages Not Received**
   - Verify subscription to correct endpoint
   - Check room filters and permissions
   - Monitor connection health with heartbeats

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('websocket_api').setLevel(logging.DEBUG)
```

### Log Analysis

Key log messages to monitor:

- Connection establishment and authentication
- Subscription management
- Message publishing and delivery
- Rate limiting events
- Error conditions and recovery

## Performance Optimization

### Client-Side

- Implement connection pooling for multiple endpoints
- Use message batching where appropriate
- Handle reconnections with exponential backoff
- Process messages asynchronously

### Server-Side

- Monitor connection counts and resource usage
- Adjust heartbeat intervals based on network conditions
- Optimize message serialization and delivery
- Use connection keep-alive for stability

## API Versioning

The WebSocket API follows semantic versioning:

- **Major version**: Breaking changes to message format
- **Minor version**: New features, backward compatible
- **Patch version**: Bug fixes, no API changes

Current version: `1.0.0`

Version information is included in connection welcome messages.

## Integration with Home Assistant

The WebSocket API seamlessly integrates with Home Assistant through the existing MQTT bridge:

1. **Predictions** → MQTT topics → Home Assistant sensors
2. **WebSocket API** → Real-time updates to external systems
3. **Alerts** → Both MQTT and WebSocket for redundancy

This dual-channel approach ensures both Home Assistant integration and external system connectivity.