#!/usr/bin/env python3

"""
GoColonyCounter.py

Colony Counter with Petri Dish Detection and HSV Brightness Masking

Author: Heekuk Park (hp2523@cumc.columbia.edu)
Version: 1.02
Date: 2025-04-02
"""

import os
import argparse
import csv
import cv2
import numpy as np
from scipy import ndimage

# --- Detect Petri Dish and Count Colonies using HSV Brightness Masking ---
def count_colonies_with_dish_detection(image_path, output_img_path=None, min_area=2, resize_max=800, brightness_threshold=190, split_plate=1):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"no image: {image_path}")

    h, w = img.shape[:2]
    scale = resize_max / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2]

    blurred = cv2.GaussianBlur(v_channel, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=200,
        param1=50, param2=35, minRadius=100, maxRadius=300
    )

    if circles is None:
        raise ValueError("Petri dish not found.")

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)

    mask = np.zeros_like(v_channel)
    cv2.circle(mask, (x, y), r, 255, -1)

    _, bright_mask = cv2.threshold(v_channel, brightness_threshold, 255, cv2.THRESH_BINARY)
    colony_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=mask)

    dist_transform = cv2.distanceTransform(colony_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(colony_mask, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), markers)

    contours = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        component_mask = np.uint8(markers == label)
        cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cnts)

    left_count = 0
    right_count = 0
    left_colonies = []
    right_colonies = []
    all_colonies = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        dist = np.sqrt((cx - x)**2 + (cy - y)**2)
        if dist > r * 0.9:
            continue

        if split_plate == 2 and abs(cx - x) < 20:
            continue

        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        aspect_ratio = max(w_ / h_, h_ / w_)
        if aspect_ratio > 4.0:
            continue

        if split_plate == 2:
            if cx < x:
                left_colonies.append((int(cx), int(cy)))
                left_count += 1
            else:
                right_colonies.append((int(cx), int(cy)))
                right_count += 1
        else:
            all_colonies.append((int(cx), int(cy)))

    if split_plate == 2:
        for center in left_colonies:
            cv2.circle(img, center, 2, (0, 0, 255), -1)
        for center in right_colonies:
            cv2.circle(img, center, 2, (255, 0, 0), -1)
        cv2.line(img, (x, y - r), (x, y + r), (255, 255, 0), 2)
        cv2.putText(img, f"L: {left_count}", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"R: {right_count}", (x + 20, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        for center in all_colonies:
            cv2.circle(img, center, 2, (0, 0, 255), -1)
        cv2.putText(img, f"Total: {len(all_colonies)}", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if output_img_path:
        cv2.imwrite(output_img_path, img)

    if split_plate == 2:
        return left_count, right_count
    else:
        return len(all_colonies),

# --- Main Batch Script ---
def main(input_dir, output_dir, split_plate, min_area, brightness_threshold):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "colony_counts.csv")

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if split_plate == 2:
            writer.writerow(["Image_Name", "Left_Colony_Count", "Right_Colony_Count"])
        else:
            writer.writerow(["Image_Name", "Colony_Count"])

        for filename in sorted(os.listdir(input_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp")):
                img_path = os.path.join(input_dir, filename)
                output_img_path = os.path.join(output_dir, f"result_{filename}")

                try:
                    counts = count_colonies_with_dish_detection(
                        img_path,
                        output_img_path=output_img_path,
                        split_plate=split_plate,
                        min_area=min_area,
                        brightness_threshold=brightness_threshold,
                    )
                    print(f"{filename} → {counts} colonies")
                    writer.writerow([filename] + list(counts))
                except Exception as e:
                    print(f"[Error] {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Colony Counter (v1.02, 2025-04-02) - Automatically count colonies in petri dish images.\n"
                    "Author: Heekuk Park <hp2523@cumc.columbia.edu>"
    )
    parser.add_argument("-i", "--input", required=True, help="Input image directory (e.g. JPG, PNG, TIF)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for results")
    parser.add_argument("-p", "--plate", type=int, choices=[1, 2], default=1,
                        help="1: Whole plate analysis (default), 2: Split plate into Left/Right")
    parser.add_argument("--min_area", type=int, default=2, help="Minimum area for colony detection (default=2)")
    parser.add_argument("--brightness_threshold", type=int, default=190, help="Threshold for bright area in HSV (default=190)")

    args = parser.parse_args()
    main(args.input, args.output, args.plate, args.min_area, args.brightness_threshold)
