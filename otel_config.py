#!/usr/bin/env python3
"""
Enhanced OpenTelemetry Configuration for Document RAG System
Provides hierarchical service tracing with proper W3C context propagation
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

# Enhanced propagation imports - W3C standard + fallbacks
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.b3 import B3MultiFormat, B3Format
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from opentelemetry.trace import Status, StatusCode


# Global state management
_initialized_services = set()
_service_hierarchy = {}

# Service configuration from environment
SERVICE_CONFIG = {
    "otlp_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://172.31.41.170:4317"),
    "environment": os.getenv("OTEL_ENVIRONMENT", "production"),
    "service_namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "document-rag-system"),
    "resource_attributes": os.getenv("OTEL_RESOURCE_ATTRIBUTES", ""),
    "batch_timeout": 2000,  # Reduced for better real-time visibility
    "batch_size": 16,       # Smaller batches for faster export
    "queue_size": 128,
}

# Enhanced Service hierarchy mapping for proper parent-child relationships
SERVICE_HIERARCHY = {
    "document-rag-orchestrator": {
        "parent": None,
        "children": ["document-rag-api", "document-rag-backend", "process-manager"],
        "type": "orchestrator"
    },
    "process-manager": {
        "parent": "document-rag-orchestrator",
        "children": ["backend-process-monitor", "api-process-monitor"],
        "type": "service_manager"
    },
    "document-rag-backend": {
        "parent": "document-rag-orchestrator",
        "children": [
            "google-drive-monitor", "local-file-scanner", "document-processor",
            "text-splitter", "embedding-generator", "vector-store-manager", 
            "file-fingerprint-db"
        ],
        "type": "processing_service"
    },
    "document-processor": {
        "parent": "document-rag-backend",
        "children": ["pdf-loader", "docx-loader", "pptx-loader", "image-processor"],
        "type": "component"
    },
    "document-rag-api": {
        "parent": "document-rag-orchestrator",
        "children": ["query-processor", "session-manager", "response-generator", "backend-proxy"],
        "type": "api_service"
    },
    # Component-level services
    "google-drive-monitor": {"parent": "document-rag-backend", "type": "component"},
    "local-file-scanner": {"parent": "document-rag-backend", "type": "component"},
    "text-splitter": {"parent": "document-rag-backend", "type": "component"},
    "embedding-generator": {"parent": "document-rag-backend", "type": "component"},
    "vector-store-manager": {"parent": "document-rag-backend", "type": "component"},
    "file-fingerprint-db": {"parent": "document-rag-backend", "type": "component"},
    "query-processor": {"parent": "document-rag-api", "type": "component"},
    "session-manager": {"parent": "document-rag-api", "type": "component"},
    "response-generator": {"parent": "document-rag-api", "type": "component"},
    "backend-proxy": {"parent": "document-rag-api", "type": "component"},
    "pdf-loader": {"parent": "document-processor", "type": "sub_component"},
    "docx-loader": {"parent": "document-processor", "type": "sub_component"},
    "pptx-loader": {"parent": "document-processor", "type": "sub_component"},
    "image-processor": {"parent": "document-processor", "type": "sub_component"},
}

def setup_enhanced_propagators():
    """
    Setup enhanced propagators following W3C standards with fallbacks
    Based on OpenTelemetry best practices for distributed tracing
    """
    # Create composite propagator with W3C as primary and fallbacks
    # This ensures maximum compatibility across different systems
    propagators = [
        TraceContextTextMapPropagator(),  # W3C Trace Context (primary)
        W3CBaggagePropagator(),          # W3C Baggage (for context data)
        B3MultiFormat(),                 # B3 Multi-header (Zipkin compatibility)
        B3Format(),                      # B3 Single-header (legacy)
        JaegerPropagator(),              # Jaeger compatibility
    ]
    
    composite_propagator = CompositePropagator(propagators)
    propagate.set_global_textmap(composite_propagator)
    
    print(f"✅ Enhanced propagators configured: W3C TraceContext (primary) + B3 + Jaeger")
    return composite_propagator

def create_enhanced_resource(service_name: str, service_version: str, environment: str) -> Resource:
    """Create OpenTelemetry resource with enhanced hierarchical service information"""
    
    # Get service hierarchy information
    hierarchy_info = SERVICE_HIERARCHY.get(service_name, {})
    parent_service = hierarchy_info.get("parent")
    service_type = hierarchy_info.get("type", "service")
    
    # Enhanced attributes following OpenTelemetry semantic conventions
    attributes = {
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": environment,
        "service.namespace": SERVICE_CONFIG["service_namespace"],
        "service.instance.id": f"{service_name}-{os.getenv('HOSTNAME', os.getpid())}",
        "service.type": service_type,
        
        # Enhanced resource attributes for better service mapping
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python",
        "telemetry.sdk.version": "1.21.0",
        
        # Deployment information
        "deployment.stage": environment,
        "deployment.name": SERVICE_CONFIG["service_namespace"],
        
        # Process information
        "process.pid": str(os.getpid()),
        "process.executable.name": "python",
        
        # Host information
        "host.name": os.getenv('HOSTNAME', 'localhost'),
    }
    
    # Add hierarchy information for service map visualization
    if parent_service:
        attributes["service.parent"] = parent_service
        attributes["service.hierarchy.level"] = _get_hierarchy_level(service_name)
    else:
        attributes["service.hierarchy.level"] = 0
        attributes["service.root"] = "true"
    
    children = hierarchy_info.get("children", [])
    if children:
        attributes["service.children"] = ",".join(children)
        attributes["service.children.count"] = str(len(children))
    
    # Add custom resource attributes from environment
    if SERVICE_CONFIG["resource_attributes"]:
        for attr in SERVICE_CONFIG["resource_attributes"].split(","):
            if "=" in attr:
                key, value = attr.strip().split("=", 1)
                attributes[key] = value
    
    return Resource.create(attributes)

def _get_hierarchy_level(service_name: str) -> int:
    """Calculate hierarchy level for service mapping"""
    level = 0
    current = service_name
    
    while current and current in SERVICE_HIERARCHY:
        parent = SERVICE_HIERARCHY[current].get("parent")
        if parent:
            level += 1
            current = parent
        else:
            break
    
    return level

def get_meter(service_name: str = None, service_version: str = "1.0.0"):
    """Get or create a meter for a specific service component with hierarchy"""
    if not service_name:
        service_name = "default"
    
    return metrics.get_meter(
        service_name,
        service_version,
        schema_url="https://opentelemetry.io/schemas/1.21.0"
    )


def initialize_opentelemetry(
    service_name: str,
    service_version: Optional[str] = None,
    environment: Optional[str] = None,
    parent_context: Optional[Any] = None
) -> Tuple[trace.Tracer, metrics.Meter]:
    """
    Initialize OpenTelemetry with enhanced W3C propagation and hierarchical service structure
    
    Args:
        service_name: Name of the service
        service_version: Version of the service  
        environment: Deployment environment
        parent_context: Parent trace context for hierarchy
        
    Returns:
        Tuple of (tracer, meter) instances
    """
    global _initialized_services
    
    # Use provided values or defaults
    current_service_version = service_version or "1.0.0"
    current_environment = environment or SERVICE_CONFIG["environment"]
    
    # Setup enhanced propagators first (critical for service map)
    setup_enhanced_propagators()
    
    # Create unique resource for this service with hierarchy
    resource = create_enhanced_resource(service_name, current_service_version, current_environment)
    
    # Check if global providers need initialization or update
    service_key = f"{service_name}-{current_environment}"
    
    if not _initialized_services:
        # Initialize global providers once with enhanced configuration
        _initialize_global_providers(resource)
    
    # Create service-specific tracer and meter
    tracer = trace.get_tracer(
        service_name, 
        current_service_version,
        schema_url="https://opentelemetry.io/schemas/1.21.0"
    )
    meter = metrics.get_meter(
        service_name, 
        current_service_version,
        schema_url="https://opentelemetry.io/schemas/1.21.0"
    )
    
    # Store initialization state
    _initialized_services.add(service_key)
    _service_hierarchy[service_name] = SERVICE_HIERARCHY.get(service_name, {})
    
    # Configure logging with enhanced correlation
    if len(_initialized_services) == 1:
        _configure_enhanced_logging()
    
    # Enhanced initialization feedback
    hierarchy_info = SERVICE_HIERARCHY.get(service_name, {})
    parent = hierarchy_info.get("parent", "root")
    service_type = hierarchy_info.get("type", "service")
    
    print(f"✅ OpenTelemetry initialized for {service_name}")
    print(f"   Type: {service_type}")
    print(f"   Parent: {parent}")
    print(f"   Environment: {current_environment}")
    print(f"   OTLP Endpoint: {SERVICE_CONFIG['otlp_endpoint']}")
    print(f"   Hierarchy Level: {_get_hierarchy_level(service_name)}")
    
    return tracer, meter

def _initialize_global_providers(resource: Resource):
    """Initialize global trace and metric providers with enhanced configuration"""
    
    # Enhanced tracing configuration
    trace_provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(1.0))  # 100% sampling for debugging
    )
    
    # OTLP Trace Exporter with enhanced configuration
    otlp_trace_exporter = OTLPSpanExporter(
        endpoint=SERVICE_CONFIG["otlp_endpoint"],
        insecure=True,
        headers={
            "Authorization": f"Bearer {os.getenv('OTEL_AUTH_TOKEN', '')}",
        } if os.getenv('OTEL_AUTH_TOKEN') else {}
    )
    
    # Enhanced batch span processor for better real-time visibility
    span_processor = BatchSpanProcessor(
        otlp_trace_exporter,
        max_queue_size=SERVICE_CONFIG["queue_size"],
        max_export_batch_size=SERVICE_CONFIG["batch_size"],
        export_timeout_millis=10000,  # 10 seconds timeout
        schedule_delay_millis=SERVICE_CONFIG["batch_timeout"]  # 2 seconds delay
    )
    
    trace_provider.add_span_processor(span_processor)
    
    # Set global trace provider
    trace.set_tracer_provider(trace_provider)
    
    # Enhanced metrics configuration
    otlp_metric_exporter = OTLPMetricExporter(
        endpoint=SERVICE_CONFIG["otlp_endpoint"],
        insecure=True,
        headers={
            "Authorization": f"Bearer {os.getenv('OTEL_AUTH_TOKEN', '')}",
        } if os.getenv('OTEL_AUTH_TOKEN') else {}
    )
    
    metric_reader = PeriodicExportingMetricReader(
        exporter=otlp_metric_exporter,
        export_interval_millis=10000  # Export every 10 seconds for better visibility
    )
    
    metric_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    )
    
    # Set global metric provider
    metrics.set_meter_provider(metric_provider)
    
    # Initialize enhanced auto-instrumentation
    _setup_enhanced_auto_instrumentation()

def _setup_enhanced_auto_instrumentation():
    """Setup enhanced automatic instrumentation for common libraries"""
    try:
        # Enhanced logging instrumentation
        LoggingInstrumentor().instrument(
            set_logging_format=True,
            log_correlation=True
        )
    except Exception:
        pass
    
    try:
        # Enhanced HTTP client instrumentation
        RequestsInstrumentor().instrument(
            excluded_urls="healthcheck,metrics"  # Exclude noise
        )
    except Exception:
        pass
    
    try:
        # Enhanced HTTPX instrumentation
        HTTPXClientInstrumentor().instrument()
    except Exception:
        pass
    
    try:
        # Enhanced FastAPI instrumentation
        FastAPIInstrumentor().instrument_app(
            excluded_urls="healthcheck,metrics,static"
        )
    except Exception:
        pass

def _configure_enhanced_logging():
    """Configure enhanced logging with trace correlation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [service=%(service_name)s trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - %(message)s'
    )

# Enhanced utility functions for service hierarchy and context propagation

def get_service_tracer(service_name: str) -> trace.Tracer:
    """Get or create a tracer for a specific service component with hierarchy"""
    if service_name not in _initialized_services:
        # Auto-initialize with proper hierarchy
        parent_info = SERVICE_HIERARCHY.get(service_name, {})
        parent_service = parent_info.get("parent")
        
        # If this service has a parent, ensure proper initialization order
        if parent_service and f"{parent_service}-{SERVICE_CONFIG['environment']}" not in _initialized_services:
            print(f"⚠️  Warning: Initializing {service_name} before parent {parent_service}")
        
        initialize_opentelemetry(service_name)
    
    return trace.get_tracer(
        service_name,
        "1.0.0",
        schema_url="https://opentelemetry.io/schemas/1.21.0"
    )

def create_child_span_with_context(
    parent_span, 
    operation_name: str, 
    service_name: str = None, 
    **attributes
):
    """Create a child span with proper hierarchy and W3C context propagation"""
    tracer = get_service_tracer(service_name) if service_name else trace.get_tracer(__name__)
    
    # Create child span with parent context
    with tracer.start_as_current_span(
        operation_name,
        kind=trace.SpanKind.INTERNAL,
        attributes=attributes
    ) as span:
        # Add enhanced service hierarchy attributes
        if service_name and service_name in SERVICE_HIERARCHY:
            hierarchy_info = SERVICE_HIERARCHY[service_name]
            parent_service = hierarchy_info.get("parent")
            service_type = hierarchy_info.get("type")
            
            if parent_service:
                span.set_attribute("service.parent", parent_service)
            if service_type:
                span.set_attribute("service.type", service_type)
            
            span.set_attribute("service.hierarchy.level", _get_hierarchy_level(service_name))
        
        return span

# Enhanced propagation utilities for HTTP calls

def inject_trace_context(headers: dict) -> dict:
    """Inject current trace context into HTTP headers using W3C standard"""
    if headers is None:
        headers = {}
    
    # Use global propagator to inject context
    propagate.inject(headers)
    
    # Add custom correlation headers
    current_span = trace.get_current_span()
    if current_span.is_recording():
        span_context = current_span.get_span_context()
        headers["X-Trace-ID"] = format(span_context.trace_id, '032x')
        headers["X-Span-ID"] = format(span_context.span_id, '016x')
    
    return headers

def extract_trace_context(headers: dict):
    """Extract trace context from HTTP headers using W3C standard"""
    if not headers:
        return None
    
    # Use global propagator to extract context
    return propagate.extract(headers)

# Enhanced tracing decorators

def traced_function(operation_name: str = None, service_name: str = None, **span_attributes):
    """Enhanced decorator for automatic function tracing with service context"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_service_tracer(service_name) if service_name else trace.get_tracer(__name__)
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(
                span_name,
                attributes=span_attributes
            ) as span:
                try:
                    # Add enhanced function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.file", func.__code__.co_filename)
                    span.set_attribute("function.line", func.__code__.co_firstlineno)
                    
                    if service_name:
                        hierarchy_info = SERVICE_HIERARCHY.get(service_name, {})
                        span.set_attribute("service.component", service_name)
                        span.set_attribute("service.type", hierarchy_info.get("type", "component"))
                        
                        parent_service = hierarchy_info.get("parent")
                        if parent_service:
                            span.set_attribute("service.parent", parent_service)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Add result metadata if available
                    if hasattr(result, '__len__'):
                        try:
                            span.set_attribute("result.length", len(result))
                        except:
                            pass
                    
                    span.set_attribute("function.status", "success")
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("function.status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator

def trace_http_call(method: str, url: str, service_name: str = None):
    """Enhanced decorator for HTTP calls with proper W3C trace propagation"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_service_tracer(service_name) if service_name else trace.get_tracer(__name__)
            
            with tracer.start_as_current_span(
                f"HTTP {method.upper()}",
                kind=trace.SpanKind.CLIENT,
                attributes={
                    "http.method": method.upper(),
                    "http.url": url,
                    "http.scheme": url.split("://")[0] if "://" in url else "http",
                    "net.peer.name": url.split("://")[1].split("/")[0] if "://" in url else url.split("/")[0]
                }
            ) as span:
                
                if service_name:
                    hierarchy_info = SERVICE_HIERARCHY.get(service_name, {})
                    span.set_attribute("service.component", service_name)
                    span.set_attribute("service.type", hierarchy_info.get("type", "component"))
                
                # Enhanced header injection with W3C propagation
                headers = kwargs.get('headers', {})
                headers = inject_trace_context(headers)
                kwargs['headers'] = headers
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Add enhanced response attributes
                    if hasattr(result, 'status_code'):
                        span.set_attribute("http.status_code", result.status_code)
                        span.set_attribute("http.status_class", f"{result.status_code // 100}xx")
                        
                        if result.status_code >= 400:
                            span.set_status(trace.Status.Error, f"HTTP {result.status_code}")
                        else:
                            span.set_status(Status(StatusCode.OK))
                    
                    if hasattr(result, 'headers'):
                        content_length = result.headers.get('content-length')
                        if content_length:
                            span.set_attribute("http.response_content_length", int(content_length))
                    
                    span.set_attribute("http.client.status", "success")
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("http.client.status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_status(trace.Status.Error, str(e))
                    raise
                    
        return wrapper
    return decorator

# Context correlation helpers

def get_current_trace_context() -> Dict[str, str]:
    """Get current trace context with enhanced information"""
    span = trace.get_current_span()
    if span and span.is_recording():
        span_context = span.get_span_context()
        return {
            "trace_id": format(span_context.trace_id, '032x'),
            "span_id": format(span_context.span_id, '016x'),
            "trace_flags": format(span_context.trace_flags, '02x'),
            "trace_state": str(span_context.trace_state) if span_context.trace_state else "",
        }
    return {"trace_id": "", "span_id": "", "trace_flags": "", "trace_state": ""}

def get_current_trace_id() -> str:
    """Get current trace ID for HTTP headers and correlation"""
    context = get_current_trace_context()
    return context.get("trace_id", "")

def get_current_span_id() -> str:
    """Get current span ID for HTTP headers and correlation"""
    context = get_current_trace_context()
    return context.get("span_id", "")

# Enhanced logging correlation

def add_trace_correlation_to_log(logger):
    """Add enhanced trace correlation to log records"""
    original_log = logger._log
    
    def correlated_log(level, msg, args, **kwargs):
        context = get_current_trace_context()
        extra = kwargs.get('extra', {})
        extra.update(context)
        
        # Add service information if available
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            resource = trace.get_tracer_provider().resource
            if resource:
                extra.update({
                    'service_name': resource.attributes.get(SERVICE_NAME, 'unknown'),
                    'service_version': resource.attributes.get(SERVICE_VERSION, 'unknown'),
                    'deployment_environment': resource.attributes.get('deployment.environment', 'unknown'),
                })
        
        kwargs['extra'] = extra
        return original_log(level, msg, args, **kwargs)
    
    logger._log = correlated_log
    return logger

# Service health check utilities with enhanced tracing

def trace_health_check(service_name: str, check_type: str = "basic"):
    """Context manager for enhanced health check tracing"""
    class HealthCheckTracer:
        def __init__(self, service_name: str, check_type: str):
            self.service_name = service_name
            self.check_type = check_type
            self.span = None
            
        def __enter__(self):
            tracer = get_service_tracer(self.service_name)
            self.span = tracer.start_span(
                f"health_check.{self.check_type}",
                kind=trace.SpanKind.INTERNAL,
                attributes={
                    "health.service": self.service_name,
                    "health.check_type": self.check_type,
                    "service.component": self.service_name
                }
            )
            
            # Add hierarchy information
            if self.service_name in SERVICE_HIERARCHY:
                hierarchy_info = SERVICE_HIERARCHY[self.service_name]
                parent = hierarchy_info.get("parent")
                if parent:
                    self.span.set_attribute("service.parent", parent)
                self.span.set_attribute("service.type", hierarchy_info.get("type", "service"))
            
            return self.span
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.span:
                if exc_type:
                    self.span.record_exception(exc_val)
                    self.span.set_attribute("health.status", "unhealthy")
                    self.span.set_status(trace.Status.Error, str(exc_val))
                else:
                    self.span.set_attribute("health.status", "healthy")
                    span.set_status(Status(StatusCode.OK))
                self.span.end()
    
    return HealthCheckTracer(service_name, check_type)

def shutdown_opentelemetry():
    """Enhanced shutdown of OpenTelemetry providers"""
    try:
        if trace.get_tracer_provider():
            # Force flush before shutdown
            if hasattr(trace.get_tracer_provider(), 'force_flush'):
                trace.get_tracer_provider().force_flush(timeout_millis=5000)
            trace.get_tracer_provider().shutdown()
        
        if metrics.get_meter_provider():
            if hasattr(metrics.get_meter_provider(), 'force_flush'):
                metrics.get_meter_provider().force_flush(timeout_millis=5000)
            metrics.get_meter_provider().shutdown()
        
        print("🛑 OpenTelemetry shutdown complete with proper cleanup")
    except Exception as e:
        print(f"⚠️  Warning during OpenTelemetry shutdown: {e}")

# Export enhanced configuration
__all__ = [
    'initialize_opentelemetry',
    'get_service_tracer',
    'get_meter', 
    'traced_function',
    'trace_http_call',
    'inject_trace_context',
    'extract_trace_context',
    'get_current_trace_context',
    'get_current_trace_id',
    'get_current_span_id',
    'add_trace_correlation_to_log',
    'trace_health_check',
    'shutdown_opentelemetry',
    'SERVICE_HIERARCHY'
]
