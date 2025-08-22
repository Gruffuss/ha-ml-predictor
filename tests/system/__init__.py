"""
System-Level Test Suite.

This package contains comprehensive system-level tests for the Home Assistant
Occupancy Prediction System, focusing on end-to-end orchestration, failure
recovery, resource management, and system stability.

Test Modules:
- test_orchestration_failure_recovery: Component isolation and failure cascading
- test_resource_constraints: Memory, CPU, and connection limit testing  
- test_lifecycle_management: Startup sequence and graceful shutdown testing
- test_long_running_stability: Extended runtime and memory leak detection
- test_error_propagation: Cross-component error handling validation

Test Categories:
- @pytest.mark.system: System-level integration tests
- @pytest.mark.slow: Long-running stability tests (may take several minutes)
- @pytest.mark.timeout(): Tests with specific timeout requirements

Usage:
    # Run all system tests
    pytest tests/system/
    
    # Run specific test category
    pytest tests/system/ -m "system and not slow"
    
    # Run long-running stability tests
    pytest tests/system/ -m "slow"
    
    # Run with verbose output
    pytest tests/system/ -v --tb=short
"""

# Test discovery will be handled by pytest automatically
# Import statements removed to avoid import errors during discovery

__all__ = [
    "TestSystemOrchestrationFailureRecovery",
    "TestSystemResourceConstraints",
    "TestSystemLifecycleManagement",
    "TestLongRunningSystemStability",
    "TestSystemErrorPropagation",
]
