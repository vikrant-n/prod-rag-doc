import os
import base64
import re
import requests
import time
from vision.base import VisionModel
from dotenv import load_dotenv

load_dotenv()

# OpenTelemetry instrumentation
try:
    import sys
    sys.path.append('..')
    from otel_config import trace_function, traced_operation
    from opentelemetry import trace, metrics
    
    # Get tracer and meter for vision module
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    
    # Define metrics
    vision_api_calls = meter.create_counter(
        "vision_api_calls_total",
        description="Total number of vision API calls"
    )
    vision_api_duration = meter.create_histogram(
        "vision_api_duration_seconds", 
        description="Duration of vision API calls"
    )
    vision_api_errors = meter.create_counter(
        "vision_api_errors_total",
        description="Total number of vision API errors"
    )
    
    OTEL_AVAILABLE = True
except ImportError:
    # Fallback if OpenTelemetry is not available
    def trace_function(name, attributes=None):
        def decorator(func):
            return func
        return decorator
    
    def traced_operation(name):
        from contextlib import nullcontext
        return nullcontext()
    
    OTEL_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VISION_MODEL = "gpt-4o-mini"
VISION_API_URL = "https://api.openai.com/v1/chat/completions"

def _strip_markdown(text: str) -> str:
    """Remove common Markdown formatting from text."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"[#>*_~]", "", text)
    return text.strip()


class OpenAIVisionModel(VisionModel):
    @trace_function("vision.analyze_image", {"component": "vision", "operation": "image_analysis"})
    def analyze_image(self, image_path: str, prompt: str) -> str:
        start_time = time.time()
        
        with traced_operation("openai_vision_analysis") as span:
            try:
                # Set span attributes
                span.set_attribute("vision.image_path", os.path.basename(image_path))
                span.set_attribute("vision.model", OPENAI_VISION_MODEL)
                span.set_attribute("vision.prompt_length", len(prompt))
                span.set_attribute("vision.api_url", VISION_API_URL)
                
                instruction = prompt + "\nRespond in plain text only. Do not use Markdown formatting."
                
                # Read and encode image
                with traced_operation("image_encoding") as encode_span:
                    try:
                        with open(image_path, "rb") as img_file:
                            img_data = img_file.read()
                            img_b64 = base64.b64encode(img_data).decode("utf-8")
                        
                        encode_span.set_attribute("image.size_bytes", len(img_data))
                        encode_span.set_attribute("image.encoded_size", len(img_b64))
                        span.set_attribute("vision.image_size_bytes", len(img_data))
                        span.add_event("image_encoded", {"size_bytes": len(img_data)})
                        
                    except Exception as e:
                        encode_span.record_exception(e)
                        span.record_exception(e)
                        span.set_attribute("vision.result", "error")
                        span.set_attribute("vision.error_type", "file_read_error")
                        if OTEL_AVAILABLE:
                            vision_api_errors.add(1, {"error_type": "file_read_error", "model": OPENAI_VISION_MODEL})
                        print(f"[OpenAI Vision API Error - File Read] {e}")
                        return "[OpenAI Vision API Error - File Read]"
                
                # Prepare API request
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": OPENAI_VISION_MODEL,
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]}
                    ],
                    "max_tokens": 512
                }
                
                span.set_attribute("vision.max_tokens", 512)
                span.add_event("api_request_prepared", {
                    "model": OPENAI_VISION_MODEL,
                    "max_tokens": 512,
                    "instruction_length": len(instruction)
                })
                
                # Make API call
                with traced_operation("openai_api_call") as api_span:
                    api_start_time = time.time()
                    response = requests.post(VISION_API_URL, headers=headers, json=data, timeout=60)
                    api_duration = time.time() - api_start_time
                    
                    api_span.set_attribute("http.method", "POST")
                    api_span.set_attribute("http.url", VISION_API_URL)
                    api_span.set_attribute("http.status_code", response.status_code)
                    api_span.set_attribute("api.duration_seconds", api_duration)
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract response content
                    content = result["choices"][0]["message"]["content"]
                    cleaned_content = _strip_markdown(content)
                    
                    api_span.set_attribute("response.content_length", len(content))
                    api_span.set_attribute("response.cleaned_length", len(cleaned_content))
                    api_span.add_event("api_response_received", {
                        "status_code": response.status_code,
                        "content_length": len(content),
                        "duration_seconds": api_duration
                    })
                
                # Record success metrics
                total_duration = time.time() - start_time
                if OTEL_AVAILABLE:
                    vision_api_calls.add(1, {"model": OPENAI_VISION_MODEL, "status": "success"})
                    vision_api_duration.record(total_duration, {"model": OPENAI_VISION_MODEL, "status": "success"})
                
                span.set_attribute("vision.result", "success")
                span.set_attribute("vision.total_duration_seconds", total_duration)
                span.set_attribute("vision.response_length", len(cleaned_content))
                span.add_event("vision_analysis_complete", {
                    "duration_seconds": total_duration,
                    "response_length": len(cleaned_content),
                    "api_duration_seconds": api_duration
                })
                
                return cleaned_content
                
            except requests.exceptions.RequestException as e:
                # Handle API request errors
                duration = time.time() - start_time
                span.record_exception(e)
                span.set_attribute("vision.result", "error")
                span.set_attribute("vision.error_type", "api_request_error")
                span.set_attribute("vision.duration_seconds", duration)
                
                if OTEL_AVAILABLE:
                    vision_api_errors.add(1, {"error_type": "api_request_error", "model": OPENAI_VISION_MODEL})
                    vision_api_duration.record(duration, {"model": OPENAI_VISION_MODEL, "status": "error"})
                
                span.add_event("vision_analysis_failed", {
                    "error_type": "api_request_error",
                    "error_message": str(e),
                    "duration_seconds": duration
                })
                
                print(f"[OpenAI Vision API Error - Request] {e}")
                return "[OpenAI Vision API Error - Request]"
                
            except Exception as e:
                # Handle other errors
                duration = time.time() - start_time
                span.record_exception(e)
                span.set_attribute("vision.result", "error")
                span.set_attribute("vision.error_type", "general_error")
                span.set_attribute("vision.duration_seconds", duration)
                
                if OTEL_AVAILABLE:
                    vision_api_errors.add(1, {"error_type": "general_error", "model": OPENAI_VISION_MODEL})
                    vision_api_duration.record(duration, {"model": OPENAI_VISION_MODEL, "status": "error"})
                
                span.add_event("vision_analysis_failed", {
                    "error_type": "general_error", 
                    "error_message": str(e),
                    "duration_seconds": duration
                })
                
                print(f"[OpenAI Vision API Error - General] {e}")
                return "[OpenAI Vision API Error - General]"
