from typing import List, Dict
from langchain_core.documents import Document
from langchain.tools import Tool
from vision.openai import OpenAIVisionModel
from dotenv import load_dotenv

load_dotenv()

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

def analyze_images_with_vision_model(images: List[Dict], prompt: str = VISION_PROMPT) -> List[Document]:
    """
    Analyze a list of images using the vision model and return LangChain Document objects.
    
    Args:
        images (List[Dict]): List of image metadata dictionaries from image_extractor
        prompt (str): Prompt to send to the vision model
        
    Returns:
        List[Document]: List of LangChain Document objects containing image analysis
    """
    vision_model = OpenAIVisionModel()
    docs = []
    
    for img in images:
        try:
            # Use the 'path' key from image_extractor
            insight = vision_model.analyze_image(img["path"], prompt)
            docs.append(Document(
                page_content=insight,
                metadata={
                    "source": img["source_doc"],
                    "file_type": img["source_type"],
                    "content_type": "image_analysis",
                    "image_path": img["path"],
                    "image_index": img["index"],
                    "image_extension": img["extension"]
                }
            ))
        except Exception as e:
            print(f"Error analyzing image {img.get('path')}: {str(e)}")
            continue
    
    return docs

vision_tool = Tool(
    name="VisionAnalyzer",
    func=lambda images, prompt=VISION_PROMPT: analyze_images_with_vision_model(images, prompt=prompt),
    description="Extracts insights from images using a pluggable vision model."
) 