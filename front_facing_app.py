import os
import math
import io
import json
import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageStat
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
def add_black_circle_background(image, expand_ratio=0.1):
    """
    Adds a black circular background behind the wheel in an image without truncating the full image.
    
    The circle is based on the tight bounding box of non-transparent (wheel) pixels,
    and its size is expanded by the provided ratio. The new canvas is sized to 
    fully contain both the original image and the circle.
    """
    image = image.convert("RGBA")
    orig_w, orig_h = image.size

    # Better bbox detection for transparent images
    img_array = np.array(image)
    alpha_channel = img_array[:, :, 3]
    non_transparent = alpha_channel > 0
    if not np.any(non_transparent):
        return image
    
    # Get actual bounding coordinates from numpy array
    rows = np.any(non_transparent, axis=1)
    cols = np.any(non_transparent, axis=0)
    if not np.any(rows) or not np.any(cols):
        return image
        
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
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

    canvas.paste(image, (int(-new_left), int(-new_top)), image)
    return canvas

# -----------------------------------------
# Helper: Get coordinates of the wheel center from bounding polygon
# -----------------------------------------
def get_wheel_center(polygon, width, height):
    xs = [p[0] * width for p in polygon]
    ys = [p[1] * height for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# -----------------------------------------
# Helper: Improved wheel angle calculation
# -----------------------------------------
def get_wheel_angle(polygon, width, height):
    """
    Calculate the angle of the wheel using PCA on the bounding polygon points.
    This handles cases where Google Vision API returns polygons in different orientations.
    """
    points = np.array([(p[0] * width, p[1] * height) for p in polygon])
    
    # Center the points
    centered_points = points - np.mean(points, axis=0)
    
    # Perform PCA
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Get the angle of the principal component
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    angle_rad = np.arctan2(principal_component[1], principal_component[0])
    
    # Convert to degrees and normalize
    angle_deg = math.degrees(angle_rad)
    
    # Adjust angle based on aspect ratio to handle wheel orientation
    # If wheel is taller than wide, we may need to adjust the angle
    if np.ptp(points[:, 1]) > np.ptp(points[:, 0]):
        angle_deg += 90
    
    return angle_deg

# -----------------------------------------
# Helper: Add enhanced drop shadow to an image for a more realistic look
# -----------------------------------------
def add_drop_shadow(image, offset=(7, 7), blur_radius=15, shadow_color=(0, 0, 0, 130)):
    """
    Creates a more realistic drop shadow with customizable opacity and blur.
    """
    total_width = image.width + abs(offset[0]) + blur_radius * 2
    total_height = image.height + abs(offset[1]) + blur_radius * 2
    shadow_image = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
    
    # Create shadow with custom opacity
    shadow = Image.new('RGBA', image.size, shadow_color)
    mask = image.split()[-1]
    shadow.putalpha(mask)
    
    shadow_x = blur_radius + max(offset[0], 0)
    shadow_y = blur_radius + max(offset[1], 0)
    shadow_image.paste(shadow, (shadow_x, shadow_y))
    
    # Apply Gaussian blur with customizable radius
    shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    image_x = blur_radius - min(offset[0], 0)
    image_y = blur_radius - min(offset[1], 0)
    shadow_image.paste(image, (image_x, image_y), image)
    
    return shadow_image

# -----------------------------------------
# Helper: Apply lighting adjustment to match rim with car lighting
# -----------------------------------------
def adjust_rim_lighting(rim_image, car_image, rim_position):
    """
    Adjusts rim brightness and contrast to match the lighting of the car
    at the wheel position.
    """
    # Extract region around wheel position
    x, y = rim_position
    sample_radius = int(min(rim_image.width, rim_image.height) * 0.8)
    
    # Ensure sample area is within car image bounds
    left = max(0, int(x - sample_radius))
    top = max(0, int(y - sample_radius))
    right = min(car_image.width, int(x + sample_radius))
    bottom = min(car_image.height, int(y + sample_radius))
    
    if right <= left or bottom <= top:
        return rim_image  # Can't sample, return original
    
    # Sample region from car image and get average brightness
    car_region = car_image.crop((left, top, right, bottom))
    car_brightness = ImageStat.Stat(car_region.convert("L")).mean[0] / 255.0
    
    # Adjust rim brightness to match
    rim_brightness = ImageStat.Stat(rim_image.convert("L")).mean[0] / 255.0
    brightness_factor = car_brightness / rim_brightness if rim_brightness > 0 else 1.0
    
    # Clamp brightness adjustment to reasonable values
    brightness_factor = max(0.7, min(1.3, brightness_factor))
    
    # Apply brightness adjustment
    enhancer = ImageEnhance.Brightness(rim_image)
    adjusted_rim = enhancer.enhance(brightness_factor)
    
    # Adjust contrast slightly to match car image
    contrast_enhancer = ImageEnhance.Contrast(adjusted_rim)
    adjusted_rim = contrast_enhancer.enhance(1.1)  # Slight contrast boost
    
    return adjusted_rim

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
# Helper: Check if a wheel is likely a front wheel
# -----------------------------------------
def is_front_wheel(polygon, width, height, all_wheels):
    """
    Determine if a wheel is likely a front wheel based on position.
    For side views, this is usually the wheel more to the left.
    """
    # Get the x-coordinate of wheel center
    wheel_center_x, _ = get_wheel_center(polygon, width, height)
    wheel_center_x_normalized = wheel_center_x / width
    
    # Get other wheel centers
    other_centers = []
    for other_poly in all_wheels:
        if other_poly != polygon:
            other_x, _ = get_wheel_center(other_poly, width, height)
            other_centers.append(other_x / width)
    
    # If no other wheels, use position heuristic
    if not other_centers:
        # Front wheels are typically on the left half for side views
        return wheel_center_x_normalized < 0.5
    
    # Otherwise, leftmost wheel is front wheel in side view
    return wheel_center_x_normalized < min(other_centers)

# -----------------------------------------
# Helper: Find coefficients for perspective transformation
# -----------------------------------------
def find_coeffs(pa, pb):
    """
    Calculate coefficients for perspective transformation.
    Source: https://stackoverflow.com/questions/14177744/
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

# -----------------------------------------
# Helper: Determine if car is in side view or angled view
# -----------------------------------------
def determine_car_view(objects, width, height):
    """
    Analyze the detected objects to determine if the car is in side view or angled view.
    Returns True for angled view, False for side view.
    """
    if not objects:
        return False  # Default to side view processing if no objects detected
    
    # Count wheels and check their positions
    wheel_polygons = []
    for obj in objects:
        name = obj.name.lower()
        if 'wheel' in name or 'tire' in name:
            polygon = []
            for vertex in obj.bounding_poly.normalized_vertices:
                polygon.append((vertex.x, vertex.y))
            wheel_polygons.append(polygon)
    
    if len(wheel_polygons) < 2:
        return False  # Not enough wheels to determine, default to side view
    
    # Get wheel centers
    wheel_centers = []
    for poly in wheel_polygons:
        center_x, center_y = get_wheel_center(poly, width, height)
        wheel_centers.append((center_x, center_y))
    
    # Calculate distances between wheel centers
    distances = []
    for i in range(len(wheel_centers)):
        for j in range(i+1, len(wheel_centers)):
            dist_x = wheel_centers[i][0] - wheel_centers[j][0]
            dist_y = wheel_centers[i][1] - wheel_centers[j][1]
            distance = math.sqrt(dist_x**2 + dist_y**2)
            distances.append(distance / width)  # Normalize by image width
    
    # Check if car is detected
    car_detected = False
    for obj in objects:
        if obj.name.lower() == 'car' or obj.name.lower() == 'vehicle':
            car_detected = True
            break
    
    # Analyze wheel positions and car detection
    if len(wheel_centers) >= 2:
        # Calculate the angle between the line connecting the two furthest wheels and the horizontal
        wheel_centers.sort(key=lambda c: c[0])  # Sort by x-coordinate
        leftmost = wheel_centers[0]
        rightmost = wheel_centers[-1]
        
        if abs(leftmost[0] - rightmost[0]) < 0.1 * width:
            # Wheels are vertically aligned (rare for car photos)
            return False
        
        angle = math.degrees(math.atan2(rightmost[1] - leftmost[1], rightmost[0] - leftmost[0]))
        
        # If the angle is small, it's likely a side view
        # If the angle is significant, it's likely an angled view
        return abs(angle) > 10
    
    # Default to side view if we can't determine
    return False

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
    
    # Request both object localization and image properties (for lighting info)
    feature_types = [
        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
        vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES)
    ]
    
    request = vision.AnnotateImageRequest(
        image=image_for_vision,
        features=feature_types
    )
    
    response = client.annotate_image(request=request)
    objects = response.localized_object_annotations

    # Determine if the car is in side view or angled view
    is_angled_view = determine_car_view(objects, width, height)
    print(f"Detected car view: {'Angled (3/4) view' if is_angled_view else 'Side view'}")
    
    # Save a debug image to visualize detections
    debug_img = original.copy()
    debug_draw = ImageDraw.Draw(debug_img)

    bounding_polygons = []
    for obj in objects:
        name = obj.name.lower()
        if 'wheel' in name or 'tire' in name:
            print(f"Detected: {obj.name} (confidence: {obj.score:.2f})")
            if obj.score < 0.6:  # Filter out low confidence detections
                print(f"Skipping low confidence detection: {obj.score:.2f}")
                continue
                
            polygon = []
            for vertex in obj.bounding_poly.normalized_vertices:
                polygon.append((vertex.x, vertex.y))
            
            # Draw detection on debug image
            points = [(p[0] * width, p[1] * height) for p in polygon]
            debug_draw.polygon(points, outline=(255, 0, 0, 255))
            
            print("Bounding polygon vertices:", polygon)
            bounding_polygons.append(polygon)
    
    # Save debug image
    debug_img.save("wheel_detections.png")
    
    if not bounding_polygons:
        # Fallback: attempt wheel detection using OpenCV if Google Vision API fails
        try:
            cv_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGBA2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                param1=100, param2=30, minRadius=30, maxRadius=200
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Convert circle to normalized polygon
                    points = []
                    for angle in range(0, 360, 45):  # 8 points around circle
                        rad = math.radians(angle)
                        px = (x + r * math.cos(rad)) / width
                        py = (y + r * math.sin(rad)) / height
                        points.append((px, py))
                    bounding_polygons.append(points)
                print(f"OpenCV fallback detected {len(circles)} potential wheels")
            
            if not bounding_polygons:
                raise Exception("No wheel/tire objects detected.")
        except Exception as e:
            print(f"OpenCV fallback also failed: {str(e)}")
            raise Exception("No wheel/tire objects detected.")

    # Filter for valid wheels based on aspect ratio
    valid_polygons = []
    # Using larger wheels by reducing the shrink factor and maintaining the aspect ratio
    shrink_factor = 0.90 if is_angled_view else 0.92  # Adjusted based on view
    narrow_factor = 1.0 if is_angled_view else 0.98   # Adjusted based on view

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
        
        # Aspect ratio check based on view type
        if is_angled_view:
            # More permissive for angled view
            if aspect_ratio < 0.7 or aspect_ratio > 2.0:
                print(f"Skipping bounding box due to aspect ratio (height/width): {aspect_ratio:.2f}")
                continue
        else:
            # Stricter for side view
            if aspect_ratio < 0.8 or aspect_ratio > 1.3:
                print(f"Skipping bounding box due to aspect ratio (height/width): {aspect_ratio:.2f}")
                continue

        valid_polygons.append((poly, left, upper, right, lower, box_width, box_height))

    # Deduplicate valid detections based on bounding box overlap (IoU)
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
    # If only one unique valid wheel eligible for replacement is detected, return the message
    if len(unique_valid) == 1 and not skip_wheels_amount_validation:
        return "only one valid wheel eligible for replacement"

    # Otherwise, process and paste each unique valid wheel
    all_polygons = [item[0] for item in unique_valid]
    
    for (poly, left, upper, right, lower, box_width, box_height) in unique_valid:
        # Make wheels larger by adjusting the scale factor
        wheel_scale_factor = 1.15 if is_angled_view else 1.10  # Adjusted based on view
        
        # Adjust shrink factors based on whether it's a front or rear wheel
        # Front wheels often appear larger in perspective
        is_front = is_front_wheel(poly, width, height, all_polygons)
        
        if is_front:
            shrink_factor = 0.90  # Changed from 0.95 to 0.90 for larger front wheels
            shadow_offset = (8, 8)  # Enhanced shadow for larger wheels
        else:
            shrink_factor = 0.92  # Changed from 0.97 to 0.92 for larger rear wheels
            shadow_offset = (6, 6)  # Enhanced shadow for larger wheels
        
        angle = get_wheel_angle(poly, width, height)
        print(f"Pasting wheel at bounding box=({left}, {upper}, {right}, {lower}) with rotation={angle:.2f}°")

        # Apply the wheel_scale_factor to make wheels larger
        new_width = int(box_width * shrink_factor * narrow_factor * wheel_scale_factor)
        new_height = int(box_height * shrink_factor * wheel_scale_factor)
        
        # Make sure we maintain aspect ratio
        wheel_aspect = wheel.width / wheel.height
        target_aspect = new_width / new_height
        
        if abs(wheel_aspect - target_aspect) > 0.2:
            # Adjust to preserve wheel's aspect ratio if too different
            if wheel_aspect > target_aspect:
                new_height = int(new_width / wheel_aspect)
            else:
                new_width = int(new_height * wheel_aspect)
        
        resized_wheel = wheel.resize((new_width, new_height), Image.LANCZOS)  # Better resizing
        
        # Add a slight perspective transform if this is a side wheel
        if abs(angle) < 10 or abs(angle - 180) < 10:
            # Side view wheel - apply slight perspective
            wheel_center_x, wheel_center_y = get_wheel_center(poly, width, height)
            if wheel_center_x < width / 2:  # Left side of image
                perspective_factor = 0.05
            else:  # Right side of image
                perspective_factor = -0.05
                
            # Create perspective transformation matrix
            width_wheel, height_wheel = resized_wheel.size
            # Define corners
            corners = [(0, 0), (width_wheel, 0), (width_wheel, height_wheel), (0, height_wheel)]
            # Define new corners with perspective
            new_corners = [
                (0, 0),
                (width_wheel, int(height_wheel * perspective_factor)),
                (width_wheel, height_wheel - int(height_wheel * perspective_factor)),
                (0, height_wheel)
            ]
            # Apply perspective transformation
            try:
                coefficient = find_coeffs(new_corners, corners)
                resized_wheel = resized_wheel.transform(
                    (width_wheel, height_wheel), 
                    Image.PERSPECTIVE, 
                    coefficient, 
                    Image.BICUBIC
                )
            except Exception as e:
                print(f"Perspective transform failed: {str(e)}. Using original wheel.")
        
        rotated_wheel = resized_wheel.rotate(angle, resample=Image.BICUBIC, expand=True)
        
        # Adjust wheel lighting to match car
        wheel_center_x, wheel_center_y = get_wheel_center(poly, width, height)
        adjusted_wheel = adjust_rim_lighting(rotated_wheel, original, (wheel_center_x, wheel_center_y))
        
        # Add enhanced drop shadow with smaller offset for more realism
        blur_radius = int(min(box_width, box_height) * 0.06)  # Increased blur radius for larger wheels
        shadowed_wheel = add_drop_shadow(
            adjusted_wheel, 
            offset=shadow_offset, 
            blur_radius=blur_radius,
            shadow_color=(0, 0, 0, 100)  # Semi-transparent shadow
        )

        center_x = left + box_width // 2
        center_y = upper + box_height // 2

        # Calculate offset based on perspective - wheels lower in the frame should be shifted down
        vertical_position = center_y / height
        vertical_offset = int(box_height * 0.05 * (vertical_position - 0.5))
        
        # For angled view, add a left offset for better positioning
        left_offset = 0
        if is_angled_view:
            left_offset = 15 if is_front else 10
        
        paste_x = center_x - shadowed_wheel.width // 2 - left_offset
        paste_y = center_y - shadowed_wheel.height // 2 + vertical_offset

        # Create a mask for blending with the original image
        if is_angled_view:
            # For angled view, use the alpha channel as mask
            original.paste(shadowed_wheel, (paste_x, paste_y), shadowed_wheel)
        else:
            # For side view, create a soft elliptical mask for better blending
            mask = Image.new('L', shadowed_wheel.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse(
                (0, 0, shadowed_wheel.width, shadowed_wheel.height),
                fill=255
            )
            mask = mask.filter(ImageFilter.GaussianBlur(5))  # Blur the edges for softer blend
            
            original.paste(shadowed_wheel, (paste_x, paste_y), shadowed_wheel)

    # Enhance the final result
    result = original.convert("RGB")
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.05)  # Slightly increase contrast for better integration
    
    return result

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
        # Try to use remove.bg API first
        try:
            wheel_img = remove_bg_from_image(wheel_resp.content)
        except Exception as e:
            print(f"Background removal failed: {str(e)}")
            # Fallback to local image if available
            try:
                wheel_img = Image.open("wheel_no_bg_2.png")
            except:
                # If fallback fails, use original image
                wheel_img = Image.open(io.BytesIO(wheel_resp.content))
        
        wheel_img = add_black_circle_background(wheel_img, expand_ratio=0.1)
        result = process_image_from_images(car_img, wheel_img, skip_wheels_amount_validation)

        if isinstance(result, str):
            return jsonify({"message": result})
        
        result_io = io.BytesIO()
        result.save(result_io, format="JPEG")
        result_io.seek(0)
        
        result.save("result.jpg", format="JPEG")
        
        return send_file(result_io, mimetype="image/jpeg")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Welcome! Use the POST /process endpoint with JSON payload containing 'car_url' and 'wheel_url'."

if __name__ == "__main__":
    app.run(debug=True)