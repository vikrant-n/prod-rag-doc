from abc import ABC, abstractmethod

class VisionModel(ABC):
    @abstractmethod
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze the given image and return insights as a string.
        Args:
            image_path (str): Path to the image file.
            prompt (str): Prompt or instructions for the vision model.
        Returns:
            str: Vision model's analysis/insights.
        """
        pass 