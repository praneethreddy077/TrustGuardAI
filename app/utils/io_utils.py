from io import BytesIO
from PIL import Image

def read_image_from_upload(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")

def bytes_io(uploaded_file):
    return BytesIO(uploaded_file.read())
