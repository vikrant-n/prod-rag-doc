#!/usr/bin/env python3
"""
Enhanced OpenTelemetry Configuration for Document RAG System
Provides hierarchical service tracing with proper context propagation
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenTelemetry imports
from opentelemetry import trace, metrics, propagate
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
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Global tracer and meter instances
_tracer_instances: Dict[str, trace.Tracer] = {}
_meter_instances: Dict[str, metrics.Meter] = {}
_initialized_services = set()

# Service configuration from environment
SERVICE_CONFIG = {
    "otlp_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://172.31.41.170:4317"),
    "environment": os.getenv("OTEL_ENVIRONMENT", "production"),
    "service_namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "document-rag-system"),
    "resource_attributes": os.getenv("OTEL_RESOURCE_ATTRIBUTES", ""),
    "batch_timeout": 5000,
    "batch_size": 32,
    "queue_size": 256,
}

# Service hierarchy mapping for proper parent-child relationships
SERVICE_HIERARCHY = {
    "document-rag-orchestrator": {
        "parent": None,
        "children": ["document-rag-api", "document-rag-backend", "process-manager"]
    },
    "process-manager": {
        "parent": "document-rag-orchestrator",
        "children": ["backend-process-monitor", "api-process-monitor"]
    },
    "document-rag-backend": {
        "parent": "document-rag-orchestrator",
        "children": [
            "google-drive-monitor", "local-file-scanner", "document-processor",
            "text-splitter", "embedding-generator", "vector-store-manager", 
            "file-fingerprint-db"
        ]
    },
    "document-processor": {
        "parent": "document-rag-backend",
        "children": ["pdf-loader", "docx-loader", "pptx-loader", "image-processor"]
    },
    "document-rag-api": {
        "parent": "document-rag-orchestrator",
        "children": ["query-processor", "session-manager", "response-generator", "backend-proxy"]
    }
}

def create_resource(service_name: str, service_version: str, environment: str) -> Resource:
    """Create OpenTelemetry resource with hierarchical service information"""
    
    # Get parent service for hierarchy
    parent_service = SERVICE_HIERARCHY.get(service_name, {}).get("parent")
    
    # Base attributes
    attributes = {
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": environment,
        "service.namespace": SERVICE_CONFIG["service_namespace"],
        "service.instance.id": f"{service_name}-{os.getenv('HOSTNAME', 'local')}",
        "service.component": service_name.split('-')[-1] if '-' in service_name else service_name,
    }
    
    # Add hierarchy information
    if parent_service:
        attributes["service.parent"] = parent_service
    
    children = SERVICE_HIERARCHY.get(service_name, {}).get("children", [])
    if children:
        attributes["service.children"] = ",".join(children)
    
    # Add custom resource attributes from environment
    if SERVICE_CONFIG["resource_attributes"]:
        for attr in SERVICE_CONFIG["resource_attributes"].split(","):
            if "=" in attr:
                key, value = attr.strip().split("=", 1)
                attributes[key] = value
    
    return Resource.create(attributes)

def setup_propagators():
    """Setup trace context propagators for cross-service communication"""
    propagate.set_global_textmap(
        CompositePropagator([
            TraceContextTextMapPropagator(),
            B3MultiFormat(),
            JaegerPropagator(),
        ])
    )

def initialize_opentelemetry(
    service_name: str,
    service_version: Optional[str] = None,
    environment: Optional[str] = None
) -> Tuple[trace.Tracer, metrics.Meter]:
    """
    Initialize OpenTelemetry with hierarchical service structure
    
    Returns:
        Tuple of (tracer, meter) instances
    """
    global _tracer_instances, _meter_instances
    
    # Use provided values or defaults
    current_service_version = service_version or "1.0.0"
    current_environment = environment or SERVICE_CONFIG["environment"]
    
    # Setup propagators for cross-service tracing
    setup_propagators()
    
    # Create unique resource for this service
    resource = create_resource(service_name, current_service_version, current_environment)
    
    # Check if global providers need initialization
    service_key = f"{service_name}-{current_environment}"
    
    if not _initialized_services:
        # Initialize global providers once
        _initialize_global_providers(resource)
    
    # Create service-specific tracer and meter
    tracer = trace.get_tracer(service_name, current_service_version)
    meter = metrics.get_meter(service_name, current_service_version)
    
    # Store instances
    _tracer_instances[service_name] = tracer
    _meter_instances[service_name] = meter
    _initialized_services.add(service_key)
    
    # Configure logging for this service
    if len(_initialized_services) == 1:
        _configure_logging()
    
    print(f"✅ OpenTelemetry initialized for {service_name}")
    print(f"   Environment: {current_environment}")
    print(f"   OTLP Endpoint: {SERVICE_CONFIG['otlp_endpoint']}")
    if service_name in SERVICE_HIERARCHY:
        parent = SERVICE_HIERARCHY[service_name].get("parent")
        if parent:
            print(f"   Parent Service: {parent}")
    
    return tracer, meter

def _initialize_global_providers(resource: Resource):
    """Initialize global trace and metric providers"""
    
    # Configure tracing
    trace_provider = TracerProvider(resource=resource)
    
    # OTLP Trace Exporter
    otlp_trace_exporter = OTLPSpanExporter(
        endpoint=SERVICE_CONFIG["otlp_endpoint"],
        insecure=True,
        headers={}
    )
    
    # Add batch span processor with environment-based configuration
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            otlp_trace_exporter,
            max_queue_size=SERVICE_CONFIG["queue_size"],
            max_export_batch_size=SERVICE_CONFIG["batch_size"],
            export_timeout_millis=30000,
            schedule_delay_millis=SERVICE_CONFIG["batch_timeout"]
        )
    )
    
    # Set global trace provider
    trace.set_tracer_provider(trace_provider)
    
    # Configure metrics
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
    
    # Set global metric provider
    metrics.set_meter_provider(metric_provider)
    
    # Initialize auto-instrumentation
    _setup_auto_instrumentation()

def _setup_auto_instrumentation():
    """Setup automatic instrumentation for common libraries"""
    try:
        LoggingInstrumentor().instrument(set_logging_format=True)
    except Exception:
        pass  # Already instrumented
    
    try:
        RequestsInstrumentor().instrument()
    except Exception:
        pass  # Already instrumented
    
    try:
        HTTPXClientInstrumentor().instrument()
    except Exception:
        pass  # Already instrumented

def _configure_logging():
    """Configure logging with trace correlation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - %(message)s'
    )

def get_tracer(service_name: Optional[str] = None) -> trace.Tracer:
    """Get tracer instance for a specific service"""
    if service_name and service_name in _tracer_instances:
        return _tracer_instances[service_name]
    
    # Don't create default - force proper initialization
    if service_name:
        # Initialize the service if not exists
        initialize_opentelemetry(service_name)
        return _tracer_instances.get(service_name, trace.get_tracer(service_name))
    
    # Return a tracer with proper name instead of default
    return trace.get_tracer("unknown-service")

def get_meter(service_name: Optional[str] = None) -> metrics.Meter:
    """Get meter instance for a specific service"""
    if service_name and service_name in _meter_instances:
        return _meter_instances[service_name]
    
    # Return default meter
    if not _meter_instances:
        initialize_opentelemetry("default-service")
    
    return list(_meter_instances.values())[0]

def get_service_tracer(service_name: str) -> trace.Tracer:
    """Get or create a tracer for a specific service component"""
    if service_name not in _tracer_instances:
        # Initialize if not exists
        initialize_opentelemetry(service_name)
    
    return _tracer_instances[service_name]

def create_child_span(parent_span, operation_name: str, service_name: str = None, **attributes):
    """Create a child span with proper hierarchy"""
    tracer = get_tracer(service_name) if service_name else get_tracer()
    
    # Create child span with parent context
    with tracer.start_as_current_span(
        operation_name,
        kind=trace.SpanKind.INTERNAL
    ) as span:
        # Add service hierarchy attributes
        if service_name and service_name in SERVICE_HIERARCHY:
            parent_service = SERVICE_HIERARCHY[service_name].get("parent")
            if parent_service:
                span.set_attribute("service.parent", parent_service)
        
        # Add custom attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        return span

def shutdown_opentelemetry():
    """Shutdown OpenTelemetry providers"""
    if trace.get_tracer_provider():
        trace.get_tracer_provider().shutdown()
    if metrics.get_meter_provider():
        metrics.get_meter_provider().shutdown()
    print("🛑 OpenTelemetry shutdown complete")

# Enhanced instrumentation decorators
def traced_function(operation_name: str = None, service_name: str = None):
    """Decorator to automatically trace function calls with service context"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_service_tracer(service_name) if service_name else get_tracer()
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    if service_name:
                        span.set_attribute("service.component", service_name)
                    
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

def trace_http_call(method: str, url: str, service_name: str = None):
    """Decorator for HTTP calls with proper trace propagation"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_service_tracer(service_name) if service_name else get_tracer()
            
            with tracer.start_as_current_span(
                f"HTTP {method.upper()}",
                kind=trace.SpanKind.CLIENT
            ) as span:
                # Add HTTP attributes
                span.set_attribute("http.method", method.upper())
                span.set_attribute("http.url", url)
                
                if service_name:
                    span.set_attribute("service.component", service_name)
                
                # Inject trace context into headers
                headers = kwargs.get('headers', {})
                propagate.inject(headers)
                kwargs['headers'] = headers
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Add response attributes if available
                    if hasattr(result, 'status_code'):
                        span.set_attribute("http.status_code", result.status_code)
                    
                    span.set_attribute("http.status", "success")
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("http.status", "error")
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

def get_current_trace_id() -> str:
    """Get current trace ID for HTTP headers"""
    context = get_trace_context()
    return context.get("trace_id", "")

def get_current_span_id() -> str:
    """Get current span ID for HTTP headers"""
    context = get_trace_context()
    return context.get("span_id", "")

# Service health check utilities
def trace_health_check(service_name: str, check_type: str = "basic"):
    """Context manager for health check tracing"""
    class HealthCheckTracer:
        def __init__(self, service_name: str, check_type: str):
            self.service_name = service_name
            self.check_type = check_type
            self.span = None
            
        def __enter__(self):
            tracer = get_service_tracer(self.service_name)
            self.span = tracer.start_span(f"health_check.{self.check_type}")
            self.span.set_attribute("health.service", self.service_name)
            self.span.set_attribute("health.check_type", self.check_type)
            return self.span
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.span:
                if exc_type:
                    self.span.record_exception(exc_val)
                    self.span.set_attribute("health.status", "unhealthy")
                else:
                    self.span.set_attribute("health.status", "healthy")
                self.span.end()
    
    return HealthCheckTracer(service_name, check_type)