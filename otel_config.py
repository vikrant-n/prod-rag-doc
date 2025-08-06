#!/usr/bin/env python3
"""
OpenTelemetry Configuration Module

Centralized configuration for OpenTelemetry instrumentation across the Document RAG system.
Provides traces, metrics, and logs with proper correlation and context propagation.
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

# OpenTelemetry Core
from opentelemetry import trace, metrics, baggage
from opentelemetry.sdk.trace import TracerProvider, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT

# OpenTelemetry Exporters
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HTTPMetricExporter

# OpenTelemetry Instrumentation
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor

# Context and Propagation
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.propagators.aws import AwsXRayPropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Global variables for shared components
_tracer_provider = None
_meter_provider = None
_tracer = None
_meter = None

class OTelConfig:
    """OpenTelemetry configuration class"""
    
    def __init__(self):
        # Service configuration
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "document-rag-system")
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
        self.environment = os.getenv("OTEL_ENVIRONMENT", os.getenv("ENVIRONMENT", "development"))
        
        # Export configuration - ensure these are set
        self.export_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.export_protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        self.export_headers = self._parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
        
        # Export to console for development
        self.console_export = os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
        
        # Sampling configuration
        self.trace_sample_rate = float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "1.0"))
        
        # Resource attributes
        self.resource_attributes = self._get_resource_attributes()
        
        # Component-specific configuration
        self.instrument_logging = os.getenv("OTEL_INSTRUMENT_LOGGING", "true").lower() == "true"
        self.instrument_requests = os.getenv("OTEL_INSTRUMENT_REQUESTS", "true").lower() == "true"
        self.instrument_sqlite = os.getenv("OTEL_INSTRUMENT_SQLITE", "true").lower() == "true"
        
        # Log correlation
        self.log_correlation = os.getenv("OTEL_PYTHON_LOG_CORRELATION", "true").lower() == "true"
        
        # Print configuration for debugging
        print(f"🔧 OpenTelemetry Config:")
        print(f"   Service: {self.service_name}")
        print(f"   Endpoint: {self.export_endpoint}")
        print(f"   Protocol: {self.export_protocol}")
        print(f"   Console Export: {self.console_export}")
    
    def _parse_headers(self, headers_str: str) -> Dict[str, str]:
        """Parse OTLP headers from environment variable"""
        headers = {}
        if headers_str:
            for header in headers_str.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    headers[key.strip()] = value.strip()
        return headers
    
    def _get_resource_attributes(self) -> Dict[str, Any]:
        """Get resource attributes for this service"""
        attributes = {
            SERVICE_NAME: self.service_name,
            SERVICE_VERSION: self.service_version,
            DEPLOYMENT_ENVIRONMENT: self.environment,
            "service.instance.id": f"{self.service_name}-{int(time.time())}",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.27.0",
            "document_rag.component": "main",
            "document_rag.version": self.service_version,
        }
        
        # Add custom attributes from environment
        custom_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        if custom_attrs:
            for attr in custom_attrs.split(","):
                if "=" in attr:
                    key, value = attr.split("=", 1)
                    attributes[key.strip()] = value.strip()
        
        return attributes

def setup_tracing(config: OTelConfig) -> trace.Tracer:
    """Set up OpenTelemetry tracing"""
    global _tracer_provider, _tracer
    
    # Create resource
    resource = Resource.create(config.resource_attributes)
    
    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)
    
    # Set up exporters
    exporters = []
    
    # Always add OTLP Exporter (not conditional)
    if config.export_protocol.lower() == "grpc":
        otlp_exporter = OTLPSpanExporter(
            endpoint=config.export_endpoint,
            headers=config.export_headers,
            insecure=True  # Set to False for production with TLS
        )
    else:
        otlp_exporter = HTTPSpanExporter(
            endpoint=config.export_endpoint,
            headers=config.export_headers
        )
    
    exporters.append(otlp_exporter)
    
    # Console exporter for development (optional)
    if config.console_export:
        exporters.append(ConsoleSpanExporter())
    
    # Add span processors
    for exporter in exporters:
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=500,
            export_timeout_millis=30000,
        )
        _tracer_provider.add_span_processor(processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    
    # Create tracer
    _tracer = trace.get_tracer(
        __name__,
        config.service_version,
        schema_url="https://opentelemetry.io/schemas/1.27.0"
    )
    
    return _tracer

def setup_metrics(config: OTelConfig) -> metrics.Meter:
    """Set up OpenTelemetry metrics"""
    global _meter_provider, _meter
    
    # Create resource
    resource = Resource.create(config.resource_attributes)
    
    # Set up metric readers
    readers = []
    
    # OTLP Metric Exporter
    if config.export_protocol.lower() == "grpc":
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=config.export_endpoint,
            headers=config.export_headers,
            insecure=True  # Set to False for production with TLS
        )
    else:
        otlp_metric_exporter = HTTPMetricExporter(
            endpoint=config.export_endpoint,
            headers=config.export_headers
        )
    
    readers.append(PeriodicExportingMetricReader(
        exporter=otlp_metric_exporter,
        export_interval_millis=10000,  # Export every 10 seconds
        export_timeout_millis=5000,
    ))
    
    # Console metric exporter for development
    if config.console_export:
        readers.append(PeriodicExportingMetricReader(
            exporter=ConsoleMetricExporter(),
            export_interval_millis=15000,  # Export every 15 seconds
        ))
    
    # Create meter provider
    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=readers
    )
    
    # Set global meter provider
    metrics.set_meter_provider(_meter_provider)
    
    # Create meter
    _meter = metrics.get_meter(
        __name__,
        config.service_version,
        schema_url="https://opentelemetry.io/schemas/1.27.0"
    )
    
    return _meter

def setup_propagation():
    """Set up context propagation"""
    # Use composite propagator for maximum compatibility
    propagators = [
        TraceContextTextMapPropagator(),  # W3C Trace Context (primary)
        B3MultiFormat(),                  # Zipkin B3 format
        JaegerPropagator(),              # Jaeger format
        AwsXRayPropagator(),             # AWS X-Ray format
    ]
    
    set_global_textmap(CompositeHTTPPropagator(propagators))

def setup_auto_instrumentation(config: OTelConfig):
    """Set up automatic instrumentation for common libraries"""
    
    # Logging instrumentation
    if config.instrument_logging:
        LoggingInstrumentor().instrument(
            set_logging_format=config.log_correlation,
            log_hook=_log_hook
        )
    
    # Requests instrumentation
    if config.instrument_requests:
        RequestsInstrumentor().instrument(
            request_hook=_requests_request_hook,
            response_hook=_requests_response_hook
        )
    
    # SQLite instrumentation
    if config.instrument_sqlite:
        SQLite3Instrumentor().instrument()

def _log_hook(span: trace.Span, record: logging.LogRecord):
    """Custom log hook to add additional context"""
    if span and span.is_recording():
        # Add trace context to log record
        span_context = span.get_span_context()
        record.trace_id = f"{span_context.trace_id:032x}"
        record.span_id = f"{span_context.span_id:016x}"
        
        # Add service information
        record.service_name = os.getenv("OTEL_SERVICE_NAME", "document-rag-system")
        record.environment = os.getenv("OTEL_ENVIRONMENT", "development")

def _requests_request_hook(span: trace.Span, request):
    """Hook for outgoing HTTP requests"""
    if span and span.is_recording():
        # Add custom attributes for document RAG system
        if hasattr(request, 'url'):
            url = str(request.url)
            if 'openai.com' in url:
                span.set_attribute("rag.external_service", "openai")
                span.set_attribute("rag.operation_type", "llm_request")
            elif 'googleapis.com' in url:
                span.set_attribute("rag.external_service", "google_drive")
                span.set_attribute("rag.operation_type", "document_fetch")

def _requests_response_hook(span: trace.Span, request, response):
    """Hook for HTTP response processing"""
    if span and span.is_recording():
        # Add response-specific attributes
        if hasattr(response, 'status_code'):
            span.set_attribute("http.response.status_code", response.status_code)
        
        # Track token usage for OpenAI requests
        if hasattr(request, 'url') and 'openai.com' in str(request.url):
            try:
                if hasattr(response, 'json'):
                    data = response.json()
                    if 'usage' in data:
                        usage = data['usage']
                        span.set_attribute("llm.token.prompt", usage.get('prompt_tokens', 0))
                        span.set_attribute("llm.token.completion", usage.get('completion_tokens', 0))
                        span.set_attribute("llm.token.total", usage.get('total_tokens', 0))
            except Exception:
                pass  # Ignore JSON parsing errors

def initialize_opentelemetry(
    service_name: Optional[str] = None,
    service_version: Optional[str] = None,
    environment: Optional[str] = None
) -> tuple[trace.Tracer, metrics.Meter]:
    """
    Initialize OpenTelemetry for the Document RAG system
    
    Args:
        service_name: Override service name
        service_version: Override service version
        environment: Override environment
    
    Returns:
        Tuple of (tracer, meter)
    """
    
    # Override config if provided
    if service_name:
        os.environ["OTEL_SERVICE_NAME"] = service_name
    if service_version:
        os.environ["OTEL_SERVICE_VERSION"] = service_version
    if environment:
        os.environ["OTEL_ENVIRONMENT"] = environment
    
    # Create configuration
    config = OTelConfig()
    
    # Set up components
    setup_propagation()
    tracer = setup_tracing(config)
    meter = setup_metrics(config)
    setup_auto_instrumentation(config)
    
    # Log initialization
    print(f"🔭 OpenTelemetry initialized for {config.service_name} v{config.service_version}")
    print(f"   Environment: {config.environment}")
    print(f"   Export endpoint: {config.export_endpoint}")
    print(f"   Export protocol: {config.export_protocol}")
    print(f"   Console export: {config.console_export}")
    print(f"   Log correlation: {config.log_correlation}")
    
    return tracer, meter

def get_tracer() -> trace.Tracer:
    """Get the global tracer instance"""
    global _tracer
    if _tracer is None:
        _tracer, _ = initialize_opentelemetry()
    return _tracer

def get_meter() -> metrics.Meter:
    """Get the global meter instance"""
    global _meter
    if _meter is None:
        _, _meter = initialize_opentelemetry()
    return _meter

def add_trace_context_to_logs():
    """Add trace context to log messages (for manual setup)"""
    class TraceContextFilter(logging.Filter):
        def filter(self, record):
            # Get current span
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                record.trace_id = f"{span_context.trace_id:032x}"
                record.span_id = f"{span_context.span_id:016x}"
                record.trace_flags = f"{span_context.trace_flags:02d}"
            else:
                record.trace_id = "0" * 32
                record.span_id = "0" * 16
                record.trace_flags = "00"
            
            record.service_name = os.getenv("OTEL_SERVICE_NAME", "document-rag-system")
            return True
    
    # Add filter to root logger
    logging.getLogger().addFilter(TraceContextFilter())

def shutdown_opentelemetry():
    """Shutdown OpenTelemetry providers gracefully"""
    global _tracer_provider, _meter_provider
    
    if _tracer_provider:
        _tracer_provider.shutdown()
        print("🔭 OpenTelemetry tracer provider shutdown complete")
    
    if _meter_provider:
        _meter_provider.shutdown()
        print("🔭 OpenTelemetry meter provider shutdown complete")

# Custom decorators for easy instrumentation
def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                if span.is_recording():
                    # Add function attributes
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    
                    # Add custom attributes
                    if attributes:
                        span.set_attributes(attributes)
                    
                    # Add argument information (be careful with sensitive data)
                    if args:
                        span.set_attribute("function.args.count", len(args))
                    if kwargs:
                        span.set_attribute("function.kwargs.count", len(kwargs))
                
                try:
                    result = func(*args, **kwargs)
                    if span.is_recording():
                        span.set_attribute("function.result.type", type(result).__name__)
                    return result
                except Exception as e:
                    if span.is_recording():
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

def trace_async_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace async function calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                if span.is_recording():
                    # Add function attributes
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    span.set_attribute("function.type", "async")
                    
                    # Add custom attributes
                    if attributes:
                        span.set_attributes(attributes)
                
                try:
                    result = await func(*args, **kwargs)
                    if span.is_recording():
                        span.set_attribute("function.result.type", type(result).__name__)
                    return result
                except Exception as e:
                    if span.is_recording():
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

# Context manager for manual spans
class traced_operation:
    """Context manager for manual span creation with automatic error handling"""
    
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.span = None
        self.tracer = get_tracer()
    
    def __enter__(self):
        self.span = self.tracer.start_span(self.name)
        self.token = trace.set_span_in_context(self.span)
        
        if self.span.is_recording():
            self.span.set_attributes(self.attributes)
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type is not None:
                self.span.record_exception(exc_val)
                self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
            else:
                self.span.set_status(trace.Status(trace.StatusCode.OK))
            
            self.span.end()

if __name__ == "__main__":
    # Test the configuration
    tracer, meter = initialize_opentelemetry(
        service_name="otel-config-test",
        environment="test"
    )
    
    # Test tracing
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("test.attribute", "test-value")
        print("✅ OpenTelemetry configuration test successful")
    
    # Shutdown
    shutdown_opentelemetry()