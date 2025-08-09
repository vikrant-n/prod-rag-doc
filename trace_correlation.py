import requests
import logging
from contextlib import contextmanager

try:
    from opentelemetry import trace
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)

class TracedSession:
    """Simple HTTP session with tracing"""
    def __init__(self, service_name: str):
        self.session = requests.Session()
        self.service_name = service_name
    
    def request(self, method: str, url: str, **kwargs):
        if not OTEL_AVAILABLE:
            return self.session.request(method, url, **kwargs)
        
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"http_{method.lower()}") as span:
            span.set_attributes({
                "http.method": method,
                "http.url": url,
                "service.name": self.service_name
            })
            
            try:
                response = self.session.request(method, url, **kwargs)
                span.set_attribute("http.status_code", response.status_code)
                return response
            except Exception as e:
                span.record_exception(e)
                raise
    
    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

@contextmanager
def traced_operation(operation_name: str):
    """Simple tracing context manager"""
    if not OTEL_AVAILABLE:
        yield
        return
    
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(operation_name) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise

# Global sessions
api_session = TracedSession("document-rag-api")
backend_session = TracedSession("document-rag-backend")