import pytesseract
from PIL import Image

def ocr_reader(image_path: str) -> str:
    """
    Extracts text from an image file using OCR.
    Useful for reading charts, scanned documents, or screenshots.
    """
    try:
        img = Image.open(image_path)
        # Simple OCR
        text = pytesseract.image_to_string(img)
        
        # Optional: specific note if text is empty
        if not text.strip():
            return "Image loaded, but no readable text found via OCR. It might be a complex diagram."
            
        return f"Text extracted from image:\n{text}"
    except Exception as e:
        return f"Error reading image: {str(e)}"