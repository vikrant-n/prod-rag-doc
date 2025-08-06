from typing import List, Dict
import time
from langchain_core.documents import Document
from langchain.tools import Tool
from vision.openai import OpenAIVisionModel
from dotenv import load_dotenv

load_dotenv()

# OpenTelemetry instrumentation
try:
    import sys
    sys.path.append('..')
    from otel_config import trace_function, traced_operation
    from opentelemetry import trace, metrics
    
    # Get tracer and meter for tools module
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    
    # Define metrics
    vision_tool_calls = meter.create_counter(
        "vision_tool_calls_total",
        description="Total number of vision tool calls"
    )
    vision_tool_duration = meter.create_histogram(
        "vision_tool_duration_seconds",
        description="Duration of vision tool processing"
    )
    vision_tool_images_processed = meter.create_counter(
        "vision_tool_images_processed_total",
        description="Total number of images processed by vision tool"
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

VISION_PROMPT = (
    "Analyze this image thoroughly and provide a detailed explanation in plain, paragraph-style text only — no lists, no tables.\n\n"
    "Your explanation should describe everything visible in the graphic, including:\n\n"
    "All identifiable visual elements such as databases, APIs, services, users, connectors, containers, data stores, people, objects, icons, text, symbols, shapes, environments, or abstract patterns.\n\n"
    "If applicable, describe the technical components (e.g., databases, APIs, services, containers) or informational structures (e.g., timelines, process flows, hierarchies, data layers).\n\n"
    "Clearly explain any flow of information, interactions, sequences, or directional cues shown by arrows, lines, or visual transitions.\n\n"
    "Describe any layout structure or visual hierarchy, such as zones, sections, columns, or tiers — and how these parts relate to each other.\n\n"
    "Point out any color-coded areas, icons, labels, or emphasis through size, boldness, or positioning — and interpret their purpose or message.\n\n"
    "Convey the overall purpose or meaning of the image — whether it's to inform, instruct, explain, advertise, narrate, or summarize.\n\n"
    "If there are charts, tables, or graphs — describe what they represent, including axes, trends, patterns, and conclusions.\n\n"
    "If there are unclear or unlabeled elements, try to infer their possible meaning based on context or common visual conventions.\n\n"
    "Avoid technical jargon unless the image itself contains it. Use natural, descriptive language to ensure clarity and accuracy.\n"
    "Your explanation should read like a thorough walkthrough — as if you’re describing the entire image to someone who can’t see it but needs to fully understand it.\n"
    "Write as much as necessary to cover every meaningful aspect of the image."
)

@trace_function("tools.analyze_images_with_vision_model", {"component": "vision_tool", "operation": "batch_image_analysis"})
def analyze_images_with_vision_model(images: List[Dict], prompt: str = VISION_PROMPT) -> List[Document]:
    """
    Analyze a list of images using the vision model and return LangChain Document objects.
    
    Args:
        images (List[Dict]): List of image metadata dictionaries from image_extractor
        prompt (str): Prompt to send to the vision model
        
    Returns:
        List[Document]: List of LangChain Document objects containing image analysis
    """
    start_time = time.time()
    
    with traced_operation("batch_image_analysis") as span:
        try:
            # Set span attributes
            span.set_attribute("vision_tool.image_count", len(images))
            span.set_attribute("vision_tool.prompt_length", len(prompt))
            span.add_event("batch_analysis_started", {
                "image_count": len(images),
                "prompt_length": len(prompt)
            })
            
            vision_model = OpenAIVisionModel()
            docs = []
            successful_analyses = 0
            failed_analyses = 0
            
            for i, img in enumerate(images):
                with traced_operation(f"image_analysis_{i}") as img_span:
                    try:
                        # Set image-specific attributes
                        img_span.set_attribute("image.index", i)
                        img_span.set_attribute("image.path", img.get("path", "unknown"))
                        img_span.set_attribute("image.source_doc", img.get("source_doc", "unknown"))
                        img_span.set_attribute("image.source_type", img.get("source_type", "unknown"))
                        img_span.set_attribute("image.extension", img.get("extension", "unknown"))
                        
                        # Use the 'path' key from image_extractor
                        insight = vision_model.analyze_image(img["path"], prompt)
                        
                        doc = Document(
                            page_content=insight,
                            metadata={
                                "source": img["source_doc"],
                                "file_type": img["source_type"],
                                "content_type": "image_analysis",
                                "image_path": img["path"],
                                "image_index": img["index"],
                                "image_extension": img["extension"]
                            }
                        )
                        docs.append(doc)
                        successful_analyses += 1
                        
                        img_span.set_attribute("analysis.result", "success")
                        img_span.set_attribute("analysis.content_length", len(insight))
                        img_span.add_event("image_analysis_success", {
                            "content_length": len(insight),
                            "image_index": i
                        })
                        
                        # Record individual image metrics
                        if OTEL_AVAILABLE:
                            vision_tool_images_processed.add(1, {
                                "status": "success",
                                "source_type": img.get("source_type", "unknown")
                            })
                        
                    except Exception as e:
                        failed_analyses += 1
                        img_span.record_exception(e)
                        img_span.set_attribute("analysis.result", "error")
                        img_span.set_attribute("analysis.error_type", type(e).__name__)
                        img_span.add_event("image_analysis_failed", {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "image_index": i
                        })
                        
                        # Record failed image metrics
                        if OTEL_AVAILABLE:
                            vision_tool_images_processed.add(1, {
                                "status": "error",
                                "source_type": img.get("source_type", "unknown")
                            })
                        
                        print(f"Error analyzing image {img.get('path')}: {str(e)}")
                        continue
            
            # Record batch metrics
            total_duration = time.time() - start_time
            if OTEL_AVAILABLE:
                vision_tool_calls.add(1, {"status": "completed"})
                vision_tool_duration.record(total_duration, {"status": "completed"})
            
            span.set_attribute("vision_tool.successful_analyses", successful_analyses)
            span.set_attribute("vision_tool.failed_analyses", failed_analyses)
            span.set_attribute("vision_tool.total_duration_seconds", total_duration)
            span.set_attribute("vision_tool.documents_created", len(docs))
            span.add_event("batch_analysis_completed", {
                "total_images": len(images),
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "documents_created": len(docs),
                "duration_seconds": total_duration
            })
            
            return docs
            
        except Exception as e:
            # Handle batch-level errors
            duration = time.time() - start_time
            span.record_exception(e)
            span.set_attribute("vision_tool.result", "error")
            span.set_attribute("vision_tool.error_type", type(e).__name__)
            span.set_attribute("vision_tool.duration_seconds", duration)
            span.add_event("batch_analysis_failed", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_seconds": duration
            })
            
            if OTEL_AVAILABLE:
                vision_tool_calls.add(1, {"status": "error"})
                vision_tool_duration.record(duration, {"status": "error"})
            
            print(f"Error in batch image analysis: {str(e)}")
            return []

vision_tool = Tool(
    name="VisionAnalyzer",
    func=lambda images, prompt=VISION_PROMPT: analyze_images_with_vision_model(images, prompt=prompt),
    description="Extracts insights from images using a pluggable vision model."
) 