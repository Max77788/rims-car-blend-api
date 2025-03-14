import os
import math
import io
import json
import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw, ImageFilter
from flask import Flask, request, send_file, jsonify
from dotenv import load_dotenv, find_dotenv
from google.cloud import vision
from google.oauth2 import service_account
import base64

if "GOOGLE_CREDENTIALS_BASE64" in os.environ:
    credentials_base64 = os.environ["GOOGLE_CREDENTIALS_BASE64"]
    credentials_json = base64.b64decode(credentials_base64).decode('utf-8')
    credentials_info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    client = vision.ImageAnnotatorClient(credentials=credentials)
else:
    client = vision.ImageAnnotatorClient()

# Load environment variables from .env file
load_dotenv(find_dotenv())

app = Flask(__name__)

# -----------------------------------------
# Configuration / File Paths & API Keys
# -----------------------------------------
API_KEY = os.environ.get("REMOVE_BG_API_KEY")
car_path = "car_image_2.png"
result_image_path = "result_image.jpg"

# -----------------------------------------
# Helper: Remove background using remove.bg API from image bytes
# -----------------------------------------
def remove_bg_from_image(image_bytes):
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': ('wheel.png', image_bytes)},
        data={'size': 'auto'},
        headers={'X-Api-Key': API_KEY},
    )
    if response.status_code == 200:
        print("Wheel background removed successfully via remove.bg.")
        return Image.open(io.BytesIO(response.content)).convert("RGBA")
    else:
        raise Exception(f"remove.bg API error: {response.status_code} {response.text}")

# -----------------------------------------
# Helper: Add black circle background to an image with transparency
# -----------------------------------------
def add_black_circle_background(image):
    image = image.convert("RGBA")
    width, height = image.size
    size = max(width, height)
    # Create a new square canvas
    bg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bg)
    # Draw a filled black circle covering the whole square
    draw.ellipse((0, 0, size, size), fill=(0, 0, 0, 255))
    # Center the wheel image over the black circle background
    x = (size - width) // 2
    y = (size - height) // 2
    bg.paste(image, (x, y), image)
    return bg

# -----------------------------------------
# Helper: Compute rotation angle from bounding polygon
# -----------------------------------------
def get_wheel_angle(bbox):
    # Expecting bbox to be a list of 4 points: [top-left, top-right, bottom-right, bottom-left]
    (x1, y1) = bbox[0]
    (x2, y2) = bbox[1]
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    # Adjust if necessary (e.g., subtract 90° if your wheel image default orientation is up)
    return angle_deg

# -----------------------------------------
# Helper: Add drop shadow to an image for a more organic look
# -----------------------------------------
def add_drop_shadow(image, offset=(5, 5), blur_radius=5):
    # Calculate new image size with extra border for the shadow
    total_width = image.width + abs(offset[0]) + blur_radius * 2
    total_height = image.height + abs(offset[1]) + blur_radius * 2
    shadow_image = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
    
    # Create a shadow from the alpha channel of the image
    shadow = Image.new('RGBA', image.size, (0, 0, 0, 255))
    mask = image.split()[-1]
    shadow.putalpha(mask)
    
    # Position the shadow within the larger image
    shadow_x = blur_radius + max(offset[0], 0)
    shadow_y = blur_radius + max(offset[1], 0)
    shadow_image.paste(shadow, (shadow_x, shadow_y))
    
    # Apply a Gaussian blur to soften the shadow edges
    shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Calculate position to paste the original image so that it overlays the shadow correctly
    image_x = blur_radius - min(offset[0], 0)
    image_y = blur_radius - min(offset[1], 0)
    shadow_image.paste(image, (image_x, image_y), image)
    
    return shadow_image

# -----------------------------------------
# Helper: Process images (car and wheel) and paste new wheels
# -----------------------------------------
def process_image_from_images(car_img, wheel_img):
    # Ensure images are in RGBA mode.
    original = car_img.convert("RGBA")
    wheel = wheel_img.convert("RGBA")
    width, height = original.size

    # Convert original image to bytes for Vision API.
    buffered = io.BytesIO()
    original.save(buffered, format="PNG")
    content = buffered.getvalue()
    image_for_vision = vision.Image(content=content)
    
    response = client.object_localization(image=image_for_vision)
    objects = response.localized_object_annotations

    # Extract bounding polygons for objects likely to be wheels/tires.
    bounding_polygons = []
    for obj in objects:
        name = obj.name.lower()
        if 'wheel' in name or 'tire' in name:
            print(f"Detected: {obj.name} (confidence: {obj.score:.2f})")
            polygon = []
            for vertex in obj.bounding_poly.normalized_vertices:
                polygon.append((vertex.x, vertex.y))
            print("Bounding polygon vertices:", polygon)
            bounding_polygons.append(polygon)
    
    if not bounding_polygons:
        raise Exception("No wheel/tire objects detected.")

    shrink_factor = 0.97  # Slightly shrink the pasted wheel image

    # For each bounding polygon, calculate paste location and rotate/resize the wheel.
    for poly in bounding_polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        pxs = [int(x * width) for x in xs]
        pys = [int(y * height) for y in ys]

        left = min(pxs)
        upper = min(pys)
        right = max(pxs)
        lower = max(pys)
        box_width = right - left
        box_height = lower - upper

        # Skip if the bounding box appears too flat compared to its width.
        if box_height < 0.8 * box_width:
            ratio = box_height / box_width
            print(f"Skipping bounding box due to short height (height/width ratio: {ratio:.2f})")
            continue

        angle = get_wheel_angle(poly)
        print(f"Pasting wheel at bounding box=({left}, {upper}, {right}, {lower}) with rotation={angle:.2f}°")

        new_width = int(box_width * shrink_factor)
        new_height = int(box_height * shrink_factor)
        
        resized_wheel = wheel.resize((new_width, new_height))
        rotated_wheel = resized_wheel.rotate(angle, resample=Image.BICUBIC, expand=True)
        
        # Add a drop shadow to create a more natural blending effect.
        shadowed_wheel = add_drop_shadow(rotated_wheel, offset=(5, 5), blur_radius=5)

        center_x = left + box_width // 2
        center_y = upper + box_height // 2

        # Paste the composite (wheel + shadow) so that its center aligns with the detected bounding box center.
        paste_x = center_x - shadowed_wheel.width // 2
        paste_y = center_y - shadowed_wheel.height // 2

        original.paste(shadowed_wheel, (paste_x, paste_y), shadowed_wheel)

    return original

# -----------------------------------------
# Flask Route: Process images from URLs
# -----------------------------------------
@app.route("/process", methods=["POST"])
def process_route():
    data = request.get_json()
    if not data or "car_url" not in data or "wheel_url" not in data:
        return jsonify({"error": "Please provide both 'car_url' and 'wheel_url' in JSON body."}), 400

    car_url = data["car_url"]
    wheel_url = data["wheel_url"]
    
    print("Car URL:", car_url)
    print("Wheel URL:", wheel_url)

    try:
        # Download car image
        car_resp = requests.get(car_url)
        car_resp.raise_for_status()
        car_img = Image.open(io.BytesIO(car_resp.content))

        # Download wheel image and remove background using remove.bg API.
        wheel_resp = requests.get(wheel_url)
        wheel_resp.raise_for_status()
        wheel_img = remove_bg_from_image(wheel_resp.content)
        
        # Add black circle background to the wheel image.
        wheel_img = add_black_circle_background(wheel_img)

        # Process the images by detecting wheels and pasting the modified wheels.
        result_img = process_image_from_images(car_img, wheel_img)

        # Save result image to in-memory file and send it.
        result_io = io.BytesIO()
        result_img.convert("RGB").save(result_io, format="JPEG")
        result_io.seek(0)
        
        result_img.convert("RGB").save("result.jpg", format="JPEG")

        return send_file(result_io, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Welcome! Use the POST /process endpoint with JSON payload containing 'car_url' and 'wheel_url'."

if __name__ == "__main__":
    app.run(debug=True)