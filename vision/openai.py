import os
import base64
import re
import requests
from vision.base import VisionModel
from dotenv import load_dotenv

load_dotenv()

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
    def analyze_image(self, image_path: str, prompt: str) -> str:
        instruction = prompt + "\nRespond in plain text only. Do not use Markdown formatting."
        with open(image_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
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
        try:
            response = requests.post(VISION_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return _strip_markdown(content)
        except Exception as e:
            print(f"[OpenAI Vision API Error] {e}")
            return "[OpenAI Vision API Error]"
