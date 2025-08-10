#!/usr/bin/env python3
"""Trace Context Middleware for W3C Propagation"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import propagate, trace
from opentelemetry.context import attach, detach

class TraceContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service_name: str = None):
        super().__init__(app)
        self.service_name = service_name
    
    async def dispatch(self, request: Request, call_next):
        # Extract W3C context from headers
        context = propagate.extract(dict(request.headers))
        token = attach(context)
        
        # Add service info to current span
        span = trace.get_current_span()
        if span.is_recording() and self.service_name:
            span.set_attribute("service.name", self.service_name)
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
        
        try:
            response = await call_next(request)
            if span.is_recording():
                span.set_attribute("http.status_code", response.status_code)
            return response
        finally:
            detach(token)
