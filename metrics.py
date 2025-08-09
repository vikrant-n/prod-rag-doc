#!/usr/bin/env python3
"""
Metrics and Monitoring for Document RAG System
Defines custom metrics for business logic monitoring
"""

import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from otel_config import get_meter
from opentelemetry import trace
from opentelemetry.metrics import Counter, Histogram, _Gauge

logger = logging.getLogger(__name__)

class RAGMetrics:
    """Document RAG specific metrics"""
    
    def __init__(self):
        self.meter = get_meter()
        
        # Document processing metrics
        self.documents_processed = self.meter.create_counter(
            name="rag.documents.processed.total",
            description="Total number of documents processed",
            unit="1"
        )
        
        self.document_processing_duration = self.meter.create_histogram(
            name="rag.document.processing.duration",
            description="Time spent processing documents",
            unit="s"
        )
        
        self.document_size = self.meter.create_histogram(
            name="rag.document.size.bytes",
            description="Size of processed documents",
            unit="bytes"
        )
        
        # Query processing metrics
        self.queries_processed = self.meter.create_counter(
            name="rag.queries.processed.total",
            description="Total number of queries processed",
            unit="1"
        )
        
        self.query_processing_duration = self.meter.create_histogram(
            name="rag.query.processing.duration",
            description="Time spent processing queries",
            unit="s"
        )
        
        self.query_relevance_score = self.meter.create_histogram(
            name="rag.query.relevance.score",
            description="Relevance score of query results",
            unit="1"
        )
        
        # Vector database metrics
        self.vector_search_duration = self.meter.create_histogram(
            name="rag.vector.search.duration",
            description="Time spent on vector searches",
            unit="s"
        )
        
        self.vector_results_count = self.meter.create_histogram(
            name="rag.vector.results.count",
            description="Number of results returned from vector search",
            unit="1"
        )
        
        # Embedding metrics
        self.embeddings_created = self.meter.create_counter(
            name="rag.embeddings.created.total",
            description="Total number of embeddings created",
            unit="1"
        )
        
        self.embedding_duration = self.meter.create_histogram(
            name="rag.embedding.duration",
            description="Time spent creating embeddings",
            unit="s"
        )
        
        # Error metrics
        self.errors_total = self.meter.create_counter(
            name="rag.errors.total",
            description="Total number of errors",
            unit="1"
        )
        
        # Active operations gauge
        self.active_operations = self.meter.create_gauge(
            name="rag.operations.active",
            description="Number of currently active operations",
            unit="1"
        )
        
        # Cache metrics
        self.cache_hits = self.meter.create_counter(
            name="rag.cache.hits.total",
            description="Total cache hits",
            unit="1"
        )
        
        self.cache_misses = self.meter.create_counter(
            name="rag.cache.misses.total",
            description="Total cache misses",
            unit="1"
        )

    @contextmanager
    def time_operation(self, operation_type: str, **attributes):
        """Context manager to time operations and record metrics"""
        start_time = time.time()
        operation_attributes = {"operation_type": operation_type, **attributes}
        
        # Increment active operations
        self.active_operations.set(1, operation_attributes)
        
        try:
            yield
        except Exception as e:
            # Record error
            error_attributes = {**operation_attributes, "error_type": type(e).__name__}
            self.errors_total.add(1, error_attributes)
            raise
        finally:
            # Record duration
            duration = time.time() - start_time
            
            if operation_type == "document_processing":
                self.document_processing_duration.record(duration, operation_attributes)
            elif operation_type == "query_processing":
                self.query_processing_duration.record(duration, operation_attributes)
            elif operation_type == "vector_search":
                self.vector_search_duration.record(duration, operation_attributes)
            elif operation_type == "embedding":
                self.embedding_duration.record(duration, operation_attributes)
            
            # Decrement active operations
            self.active_operations.set(0, operation_attributes)

    def record_document_processed(self, file_type: str, file_size: int, chunk_count: int):
        """Record document processing metrics"""
        attributes = {
            "file_type": file_type,
            "processing_status": "success"
        }
        
        self.documents_processed.add(1, attributes)
        self.document_size.record(file_size, attributes)
        
        # Record chunks created
        chunk_attributes = {**attributes, "chunk_count": str(chunk_count)}
        self.embeddings_created.add(chunk_count, chunk_attributes)

    def record_query_processed(self, query_type: str, result_count: int, confidence: float):
        """Record query processing metrics"""
        attributes = {
            "query_type": query_type,
            "processing_status": "success"
        }
        
        self.queries_processed.add(1, attributes)
        self.vector_results_count.record(result_count, attributes)
        self.query_relevance_score.record(confidence, attributes)

    def record_cache_event(self, cache_type: str, hit: bool):
        """Record cache hit/miss"""
        attributes = {"cache_type": cache_type}
        
        if hit:
            self.cache_hits.add(1, attributes)
        else:
            self.cache_misses.add(1, attributes)

# Global metrics instance
rag_metrics = RAGMetrics()

# Convenience functions
def time_document_processing(**attributes):
    """Time document processing operations"""
    return rag_metrics.time_operation("document_processing", **attributes)

def time_query_processing(**attributes):
    """Time query processing operations"""
    return rag_metrics.time_operation("query_processing", **attributes)

def time_vector_search(**attributes):
    """Time vector search operations"""
    return rag_metrics.time_operation("vector_search", **attributes)

def time_embedding(**attributes):
    """Time embedding operations"""
    return rag_metrics.time_operation("embedding", **attributes)

def record_document_processed(file_type: str, file_size: int, chunk_count: int):
    """Record document processing metrics"""
    rag_metrics.record_document_processed(file_type, file_size, chunk_count)

def record_query_processed(query_type: str, result_count: int, confidence: float):
    """Record query processing metrics"""
    rag_metrics.record_query_processed(query_type, result_count, confidence)

def record_cache_event(cache_type: str, hit: bool):
    """Record cache hit/miss"""
    rag_metrics.record_cache_event(cache_type, hit)
