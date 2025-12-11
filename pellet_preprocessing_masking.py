import cv2
import numpy as np
import os
import glob

def process_pellet_images(input_dir, output_dir):
    """
    1. Grayscale & Rough Crop (400x400) around fixed point (2736, 1824).
    2. Detect 'True' Center AND Radius using Hough Circles or Contours.
    3. Apply Circular Mask at detected location (Background -> White).
    4. Apply Manual Offsets to the CROP center (not the mask).
    5. Final Crop (300x300) centered on the offset point.
    """
    
    # --- CONFIGURATION ---
    FIXED_CENTER_X = 2736
    FIXED_CENTER_Y = 1824
    ROUGH_SIZE = 400
    FINAL_SIZE = 300
    
    # --- MANUAL OFFSETS ---
    # Use these to nudge the final frame if the pellet looks off-center.
    # OFFSET_Y = 20  -> Moves the crop DOWN, making the pellet appear HIGHER in the output.
    # OFFSET_Y = -20 -> Moves the crop UP, making the pellet appear LOWER in the output.
    OFFSET_X = -5   
    OFFSET_Y = 0

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of all images
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_files:
        print(f"No images found in {input_dir}. Please check your folder path.")
        return

    print(f"Found {len(image_files)} images. Starting processing with MASKING & OFFSET...")

    for file_path in image_files:
        try:
            # 1. Load the image
            img = cv2.imread(file_path)
            if img is None:
                continue

            # 2. Convert to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_h, img_w = gray.shape

            # --- STAGE 1: ROUGH CROP (400x400) ---
            rough_x1 = FIXED_CENTER_X - (ROUGH_SIZE // 2)
            rough_y1 = FIXED_CENTER_Y - (ROUGH_SIZE // 2)
            rough_x2 = rough_x1 + ROUGH_SIZE
            rough_y2 = rough_y1 + ROUGH_SIZE

            rough_crop = np.zeros((ROUGH_SIZE, ROUGH_SIZE), dtype=np.uint8)

            src_x1 = max(0, rough_x1)
            src_y1 = max(0, rough_y1)
            src_x2 = min(img_w, rough_x2)
            src_y2 = min(img_h, rough_y2)

            dst_x1 = max(0, -rough_x1)
            dst_y1 = max(0, -rough_y1)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            if src_x2 > src_x1 and src_y2 > src_y1:
                rough_crop[dst_y1:dst_y2, dst_x1:dst_x2] = gray[src_y1:src_y2, src_x1:src_x2]
            else:
                print(f"Skipping {os.path.basename(file_path)}: Out of bounds.")
                continue

            # --- STAGE 2: DETECTION ---
            blurred = cv2.GaussianBlur(rough_crop, (9, 9), 2)
            
            detected_center_x = ROUGH_SIZE // 2
            detected_center_y = ROUGH_SIZE // 2
            detected_radius = 0
            method_used = "None"

            # Method A: Hough Circles
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=ROUGH_SIZE/4,
                                     param1=100, param2=30, minRadius=20, maxRadius=0)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                local_center = ROUGH_SIZE // 2
                best_circle = None
                min_dist = float('inf')

                for (cx, cy, r) in circles:
                    dist = np.sqrt((cx - local_center)**2 + (cy - local_center)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_circle = (cx, cy, r)

                if best_circle:
                    detected_center_x, detected_center_y, detected_radius = best_circle
                    method_used = "Hough Circle"

            # Method B: Contours (Fallback)
            if method_used == "None":
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                    detected_center_x = int(cx)
                    detected_center_y = int(cy)
                    detected_radius = int(radius)
                    method_used = "Contour (Enclosing Circle)"

            # --- STAGE 3: MASKING (Make background white) ---
            # IMPORTANT: We mask at the DETECTED location to preserve the pellet pixels.
            if detected_radius > 0:
                masked_img = np.full((ROUGH_SIZE, ROUGH_SIZE), 255, dtype=np.uint8)
                mask = np.zeros((ROUGH_SIZE, ROUGH_SIZE), dtype=np.uint8)
                
                # Draw mask at the DETECTED center
                cv2.circle(mask, (detected_center_x-5, detected_center_y), detected_radius-65, 255, -1)
                
                masked_img[mask == 255] = rough_crop[mask == 255]
                rough_crop = masked_img

            # --- STAGE 4: CALCULATE CROP CENTER (With Offset) ---
            # We shift the CROP center, not the pellet itself.
            crop_center_x = detected_center_x + OFFSET_X
            crop_center_y = detected_center_y + OFFSET_Y

            # --- STAGE 5: FINAL CROP (300x300) ---
            final_canvas = np.full((FINAL_SIZE, FINAL_SIZE), 255, dtype=np.uint8) 
            
            half_final = FINAL_SIZE // 2
            src_x1 = int(crop_center_x - half_final)
            src_y1 = int(crop_center_y - half_final)
            src_x2 = int(crop_center_x + half_final)
            src_y2 = int(crop_center_y + half_final)

            dst_x1 = 0
            dst_y1 = 0
            dst_x2 = FINAL_SIZE
            dst_y2 = FINAL_SIZE

            if src_x1 < 0:
                dst_x1 = abs(src_x1)
                src_x1 = 0
            if src_y1 < 0:
                dst_y1 = abs(src_y1)
                src_y1 = 0
            if src_x2 > ROUGH_SIZE:
                overlap = src_x2 - ROUGH_SIZE
                dst_x2 = FINAL_SIZE - overlap
                src_x2 = ROUGH_SIZE
            if src_y2 > ROUGH_SIZE:
                overlap = src_y2 - ROUGH_SIZE
                dst_y2 = FINAL_SIZE - overlap
                src_y2 = ROUGH_SIZE

            if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                final_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = rough_crop[src_y1:src_y2, src_x1:src_x2]
                
                filename = os.path.basename(file_path)
                save_path = os.path.join(output_dir, "proc_" + filename)
                cv2.imwrite(save_path, final_canvas)
                print(f"Processed: {filename} | {method_used} (r={detected_radius}) | Offset Applied")
            else:
                print(f"Error: Final crop invalid for {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    INPUT_FOLDER = 'raw_images'
    OUTPUT_FOLDER = 'processed_images'
    
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created '{INPUT_FOLDER}' folder. Please put your pellet photos inside it and run this script again.")
    else:
        process_pellet_images(INPUT_FOLDER, OUTPUT_FOLDER)