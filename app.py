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

# Load environment variables from .env file
load_dotenv(find_dotenv())

if "GOOGLE_CREDENTIALS_BASE64" in os.environ:
    credentials_base64 = os.environ["GOOGLE_CREDENTIALS_BASE64"]
    credentials_json = base64.b64decode(credentials_base64).decode('utf-8')
    credentials_info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    client = vision.ImageAnnotatorClient(credentials=credentials)
else:
    client = vision.ImageAnnotatorClient()

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
        Image.open(io.BytesIO(response.content)).convert("RGBA").save("wheel_no_bg_2.png", format="PNG")
        return Image.open(io.BytesIO(response.content)).convert("RGBA")
    else:
        raise Exception(f"remove.bg API error: {response.status_code} {response.text}")

# -----------------------------------------
# Helper: Add black circle background to an image with transparency
# -----------------------------------------
def add_black_circle_background(image, expand_ratio=0):
    """
    Adds a black circular background behind the wheel in an image without truncating the full image.
    
    The circle is based on the tight bounding box of non-transparent (wheel) pixels,
    and its size is expanded by the provided ratio. The new canvas is sized to 
    fully contain both the original image and the circle.
    """
    image = image.convert("RGBA")
    orig_w, orig_h = image.size

    bbox = image.getbbox()  # (left, top, right, bottom)
    if not bbox:
        return image
    left, top, right, bottom = bbox
    wheel_width = right - left
    wheel_height = bottom - top

    wheel_center_x = left + wheel_width / 2
    wheel_center_y = top + wheel_height / 2

    base_size = max(wheel_width, wheel_height)
    expand_pixels = int(base_size * expand_ratio)
    circle_diameter = base_size + 2 * expand_pixels

    circle_left = wheel_center_x - circle_diameter / 2
    circle_top = wheel_center_y - circle_diameter / 2
    circle_right = wheel_center_x + circle_diameter / 2
    circle_bottom = wheel_center_y + circle_diameter / 2

    new_left = min(0, circle_left)
    new_top = min(0, circle_top)
    new_right = max(orig_w, circle_right)
    new_bottom = max(orig_h, circle_bottom)
    new_width = int(math.ceil(new_right - new_left))
    new_height = int(math.ceil(new_bottom - new_top))

    canvas = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    adj_circle_left = circle_left - new_left
    adj_circle_top = circle_top - new_top
    adj_circle_right = circle_right - new_left
    adj_circle_bottom = circle_bottom - new_top

    draw = ImageDraw.Draw(canvas)
    draw.ellipse((adj_circle_left, adj_circle_top, adj_circle_right, adj_circle_bottom),
                 fill=(0, 0, 0, 255))

    canvas.paste(image, (-new_left, -new_top), image)
    return canvas

# -----------------------------------------
# Helper: Compute rotation angle from bounding polygon
# -----------------------------------------
def get_wheel_angle(bbox):
    # Get all vertices
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    
    # Find the center of the bounding box
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    
    # Calculate angles from center to each vertex
    angles = []
    for x, y in zip(xs, ys):
        dx = x - center_x
        dy = y - center_y
        angle = math.degrees(math.atan2(dy, dx))
        angles.append(angle)
    
    # Sort angles and find the primary axis
    angles.sort()
    
    # Calculate mean angle difference between consecutive vertices (ideally ~90° for a rectangle)
    angle_diffs = [(angles[(i+1) % len(angles)] - angles[i]) % 360 for i in range(len(angles))]
    avg_angle = sum(angle_diffs) / len(angle_diffs)
    
    # Return angle adjusted to correct orientation
    return (min(angles) + avg_angle / 2) % 90

# -----------------------------------------
# Helper: Add drop shadow to an image for a more organic look
# -----------------------------------------
def add_drop_shadow(image, offset=(5, 5), blur_radius=5):
    total_width = image.width + abs(offset[0]) + blur_radius * 2
    total_height = image.height + abs(offset[1]) + blur_radius * 2
    shadow_image = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
    
    shadow = Image.new('RGBA', image.size, (0, 0, 0, 255))
    mask = image.split()[-1]
    shadow.putalpha(mask)
    
    shadow_x = blur_radius + max(offset[0], 0)
    shadow_y = blur_radius + max(offset[1], 0)
    shadow_image.paste(shadow, (shadow_x, shadow_y))
    
    shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    image_x = blur_radius - min(offset[0], 0)
    image_y = blur_radius - min(offset[1], 0)
    shadow_image.paste(image, (image_x, image_y), image)
    
    return shadow_image

# -----------------------------------------
# Helper: Compute Intersection over Union (IoU) between two boxes.
# Each box is a tuple: (left, upper, right, lower)
# -----------------------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(areaA + areaB - interArea) if (areaA + areaB - interArea) != 0 else 0
    return iou

# -----------------------------------------
# Helper: Determine if a wheel is on the left or right side of the car
# -----------------------------------------
def determine_wheel_position(wheel_center_x, image_width, wheel_center_y, image_height):
    """
    Determines if a wheel is on the left or right side of the car.
    Optimized for both frontal and side-profile images.
    
    Args:
        wheel_center_x: X-coordinate of the wheel center
        image_width: Width of the car image
        wheel_center_y: Y-coordinate of the wheel center
        image_height: Height of the car image
        
    Returns:
        position: "left" or "right"
        offset_direction: offset pixels (negative = left, positive = right)
    """
    # Check if image is likely a side profile (based on wheel positions)
    if wheel_center_x < image_width * 0.25 or wheel_center_x > image_width * 0.75:
        # Likely a side profile - determine which side based on wheel position
        if wheel_center_x < image_width / 2:
            # Wheel is on left side of image
            return "left", 15  # Offset more for side view
        else:
            # Wheel is on right side of image
            return "right", -15  # Offset more for side view
    else:
        # Likely a front/angled view - use standard logic
        if wheel_center_x < image_width / 2:
            return "left", 12  # Left wheel, offset to the right
        else:
            return "right", -12  # Right wheel, offset to the left

# -----------------------------------------
# Helper: Process images (car and wheel) and paste new wheels
# -----------------------------------------
def process_image_from_images(car_img, wheel_img, skip_wheels_amount_validation=False):
    original = car_img.convert("RGBA")
    wheel = wheel_img.convert("RGBA")
    width, height = original.size

    buffered = io.BytesIO()
    original.save(buffered, format="PNG")
    content = buffered.getvalue()
    image_for_vision = vision.Image(content=content)
    
    response = client.object_localization(image=image_for_vision)
    objects = response.localized_object_annotations

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

    # Filter for valid (square-like) wheels using an aspect ratio check.
    valid_polygons = []
    shrink_factor = 0.97
    narrow_factor = 0.95

    # Detect if this is likely a side profile image
    is_side_profile = False
    wheel_centers_x = []
    
    for poly in bounding_polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        pxs = [int(x * width) for x in xs]
        pys = [int(y * height) for y in ys]

        left = min(pxs)
        upper = min(pys)
        right = max(pxs)
        lower = max(pys)
        center_x = (left + right) // 2
        wheel_centers_x.append(center_x)
        
        # Check if wheels are mostly on the sides, indicating side profile
        if center_x < width * 0.25 or center_x > width * 0.75:
            is_side_profile = True
        
    # Adjust parameters for side profile
    if is_side_profile:
        print("Side profile detected - adjusting parameters")
        # For side profile, adjust the narrow factor to make wheels more realistic
        narrow_factor = 1.0  # No narrowing for side view
    
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

        aspect_ratio = box_height / box_width if box_width else 0
        # For side profile, allow a different aspect ratio range
        min_ratio = 0.7 if is_side_profile else 0.8
        max_ratio = 2.0 if is_side_profile else 1.8
        
        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
            print(f"Skipping bounding box due to aspect ratio (height/width): {aspect_ratio:.2f}")
            continue

        valid_polygons.append((poly, left, upper, right, lower, box_width, box_height))

    # Deduplicate valid detections based on bounding box overlap (IoU).
    unique_valid = []
    iou_threshold = 0.5
    for item in valid_polygons:
        _, left, upper, right, lower, bw, bh = item
        box = (left, upper, right, lower)
        duplicate = False
        for existing in unique_valid:
            ex_box = (existing[1], existing[2], existing[3], existing[4])
            if compute_iou(box, ex_box) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            unique_valid.append(item)
    
    print("Unique valid wheels detected:", len(unique_valid))
    # If only one unique valid wheel eligible for replacement is detected, return the message.
    if len(unique_valid) == 1 and not skip_wheels_amount_validation:
        return "only one valid wheel eligible for replacement"

    # Otherwise, process and paste each unique valid wheel.
    for (poly, left, upper, right, lower, box_width, box_height) in unique_valid:
        angle = get_wheel_angle(poly)
        center_x = left + box_width // 2
        center_y = upper + box_height // 2
        
        # Determine if wheel is on left or right side and get offset
        wheel_position, offset_x = determine_wheel_position(center_x, width, center_y, height)
        print(f"Wheel at ({center_x}, {center_y}) detected as {wheel_position} wheel. Applying {offset_x}px offset.")
        
        print(f"Pasting wheel at bounding box=({left}, {upper}, {right}, {lower}) with rotation={angle:.2f}°")

        # For side profile images, adjust wheel size
        side_factor = 1.0
        if is_side_profile:
            # Slightly larger wheels for side profile
            side_factor = 1.05
            
            # If the car is viewed directly from the side, wheels appear more
            # elliptical rather than circular. Adjust narrow_factor accordingly.
            if wheel_position == "left" and center_x < width * 0.25:
                narrow_factor = 0.65  # More narrowing for left wheel in left-side view
            elif wheel_position == "right" and center_x > width * 0.75:
                narrow_factor = 0.65  # More narrowing for right wheel in right-side view

        new_width = int(box_width * shrink_factor * narrow_factor * side_factor)
        new_height = int(box_height * shrink_factor * side_factor)
        
        resized_wheel = wheel.resize((new_width, new_height))
        rotated_wheel = resized_wheel.rotate(angle, resample=Image.BICUBIC, expand=True)
        
        shadowed_wheel = add_drop_shadow(rotated_wheel, offset=(5, 5), blur_radius=5)

        # Apply the horizontal offset based on wheel position
        paste_x = center_x - shadowed_wheel.width // 2 + offset_x
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
    skip_wheels_amount_validation = data.get("skip_wheels_amount_validation", "false").lower() == "true"
    
    print("Car URL:", car_url)
    print("Wheel URL:", wheel_url)

    try:
        car_resp = requests.get(car_url)
        car_resp.raise_for_status()
        car_img = Image.open(io.BytesIO(car_resp.content))

        wheel_resp = requests.get(wheel_url)
        wheel_resp.raise_for_status()
        # Uncomment below to use remove.bg API:
        # wheel_img = remove_bg_from_image(wheel_resp.content)
        wheel_img = Image.open("wheel_no_bg_2.png")
        
        wheel_img = add_black_circle_background(wheel_img)
        result = process_image_from_images(car_img, wheel_img, skip_wheels_amount_validation)

        if isinstance(result, str):
            return jsonify({"message": result})
        
        result_io = io.BytesIO()
        result.convert("RGB").save(result_io, format="JPEG")
        result_io.seek(0)
        
        result.convert("RGB").save("result.jpg", format="JPEG")
        
        return send_file(result_io, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Welcome! Use the POST /process endpoint with JSON payload containing 'car_url' and 'wheel_url'."

if __name__ == "__main__":
    app.run(debug=True)