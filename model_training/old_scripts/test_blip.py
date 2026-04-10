from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

print("start")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("some_test_image.jpg").convert("RGB")

inputs = processor(images=image, return_tensors="pt")
outputs = model.generate(**inputs)

caption = processor.decode(outputs[0], skip_special_tokens=True)
print("caption:", caption)
