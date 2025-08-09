#!/usr/bin/env python3
"""
OpenTelemetry Configuration for Document RAG System
Provides centralized configuration for traces, metrics, and logs
"""

import os
import logging
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Global tracer and meter instances
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None
_initialized_services = set()

# Service configuration
SERVICE_CONFIG = {
    "service_name": os.getenv("OTEL_SERVICE_NAME", "document-rag"),
    "service_version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
    "environment": os.getenv("OTEL_ENVIRONMENT", "development"),
    "otlp_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
}

def create_resource(service_name: str, service_version: str, environment: str) -> Resource:
    """Create OpenTelemetry resource with service information"""
    return Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": environment,
        "service.namespace": "document-rag-system",
        "service.instance.id": f"{service_name}-{os.getenv('HOSTNAME', 'local')}",
        "service.component": service_name.split('-')[-1] if '-' in service_name else service_name,
    })

def initialize_opentelemetry(
    service_name: Optional[str] = None,
    service_version: Optional[str] = None,
    environment: Optional[str] = None
) -> Tuple[trace.Tracer, metrics.Meter]:
    """
    Initialize OpenTelemetry with traces, metrics, and logs for each service
    
    Returns:
        Tuple of (tracer, meter) instances
    """
    global _tracer, _meter
    
    # Use provided values or defaults
    current_service_name = service_name or SERVICE_CONFIG["service_name"]
    current_service_version = service_version or SERVICE_CONFIG["service_version"]
    current_environment = environment or SERVICE_CONFIG["environment"]
    
    # Create unique resource for this service
    resource = create_resource(current_service_name, current_service_version, current_environment)
    
    # Check if this specific service has been initialized
    service_key = f"{current_service_name}-{current_environment}"
    
    if service_key not in _initialized_services:
        # Configure tracing with service-specific resource
        trace_provider = TracerProvider(resource=resource)
        
        # OTLP Trace Exporter
        otlp_trace_exporter = OTLPSpanExporter(
            endpoint=SERVICE_CONFIG["otlp_endpoint"],
            insecure=True,
            headers={}
        )
        
        # Add batch span processor
        trace_provider.add_span_processor(
            BatchSpanProcessor(
                otlp_trace_exporter,
                max_queue_size=512,
                max_export_batch_size=64,
                export_timeout_millis=30000,
                schedule_delay_millis=1000
            )
        )
        
        # Set global trace provider (only for the first service)
        if not _initialized_services:
            trace.set_tracer_provider(trace_provider)
        
        # Configure metrics with service-specific resource
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=SERVICE_CONFIG["otlp_endpoint"],
            insecure=True,
            headers={}
        )
        
        metric_reader = PeriodicExportingMetricReader(
            exporter=otlp_metric_exporter,
            export_interval_millis=30000
        )
        
        metric_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        
        # Set global metric provider (only for the first service)
        if not _initialized_services:
            metrics.set_meter_provider(metric_provider)
        
        # Initialize auto-instrumentation only once
        if not _initialized_services:
            try:
                LoggingInstrumentor().instrument(set_logging_format=True)
            except Exception:
                pass  # Already instrumented
            
            try:
                RequestsInstrumentor().instrument()
            except Exception:
                pass  # Already instrumented
        
        _initialized_services.add(service_key)
    
    # Get tracer and meter instances for this specific service
    _tracer = trace.get_tracer(current_service_name, current_service_version)
    _meter = metrics.get_meter(current_service_name, current_service_version)
    
    # Configure logging only once
    if len(_initialized_services) == 1:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - %(message)s'
        )
    
    print(f"✅ OpenTelemetry initialized for {current_service_name}")
    print(f"   Environment: {current_environment}")
    print(f"   OTLP Endpoint: {SERVICE_CONFIG['otlp_endpoint']}")
    
    return _tracer, _meter

def get_tracer() -> trace.Tracer:
    """Get the global tracer instance"""
    if _tracer is None:
        initialize_opentelemetry()
    return _tracer

def get_meter() -> metrics.Meter:
    """Get the global meter instance"""
    if _meter is None:
        initialize_opentelemetry()
    return _meter

def shutdown_opentelemetry():
    """Shutdown OpenTelemetry providers"""
    if trace.get_tracer_provider():
        trace.get_tracer_provider().shutdown()
    if metrics.get_meter_provider():
        metrics.get_meter_provider().shutdown()
    print("🛑 OpenTelemetry shutdown complete")

# Instrumentation decorators
def traced_function(operation_name: str = None):
    """Decorator to automatically trace function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Add result metadata if available
                    if hasattr(result, '__len__'):
                        span.set_attribute("result.length", len(result))
                    
                    span.set_attribute("function.status", "success")
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("function.status", "error")
                    span.set_attribute("error.message", str(e))
                    raise
        return wrapper
    return decorator

# Context correlation helpers
def get_trace_context():
    """Get current trace context for correlation"""
    span = trace.get_current_span()
    if span:
        return {
            "trace_id": format(span.get_span_context().trace_id, '032x'),
            "span_id": format(span.get_span_context().span_id, '016x')
        }
    return {"trace_id": "", "span_id": ""}

def add_trace_correlation_to_log(logger):
    """Add trace correlation to log records"""
    original_log = logger._log
    
    def correlated_log(level, msg, args, **kwargs):
        context = get_trace_context()
        extra = kwargs.get('extra', {})
        extra.update(context)
        kwargs['extra'] = extra
        return original_log(level, msg, args, **kwargs)
    
    logger._log = correlated_log
    return logger