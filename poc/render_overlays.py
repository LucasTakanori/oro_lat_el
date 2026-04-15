import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
SUBJECT = "Lucas"
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / SUBJECT
OUT_DIR = ROOT_DIR / "poc" / "overlays" / SUBJECT

OUT_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe Face Mesh Lip Indices
# Outer Lip: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
# Inner Lip: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

def process_video(video_path: Path):
    landmarks_path = video_path.with_name(video_path.stem + "_landmarks.json")
    if not landmarks_path.exists():
        print(f"Skipping {video_path.name}: Landmarks file not found.")
        return

    # Load Landmarks
    try:
        with open(landmarks_path, 'r') as f:
            data = json.load(f)
            landmarks_frames = data.get('landmarks', [])
    except Exception as e:
        print(f"Error loading landmarks for {video_path.name}: {e}")
        return

    # Open Video to get timestamps (for VFR correction)
    timestamps = []
    cap_ts = cv2.VideoCapture(str(video_path))
    if cap_ts.isOpened():
        while True:
            ret, _ = cap_ts.read()
            if not ret: break
            timestamps.append(cap_ts.get(cv2.CAP_PROP_POS_MSEC))
        cap_ts.release()

    # Open Video for processing
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path.name}")
        return

    # Video Properties
    if len(timestamps) > 1:
        duration_sec = (timestamps[-1] - timestamps[0]) / 1000.0
        fps = (len(timestamps) - 1) / duration_sec if duration_sec > 0 else 30
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup Output Video
    output_path = OUT_DIR / f"{video_path.stem}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Processing {video_path.name} -> {output_path.name}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(landmarks_frames):
            lm = landmarks_frames[frame_idx]['lm']
            
            # Draw Lip Contours Only
            pts_outer = np.array([[lm[i]['x']*width, lm[i]['y']*height] for i in OUTER_LIP if i < len(lm)], np.int32)
            if len(pts_outer) > 0:
                cv2.polylines(frame, [pts_outer], True, (0, 255, 0), 2)
            
            pts_inner = np.array([[lm[i]['x']*width, lm[i]['y']*height] for i in INNER_LIP if i < len(lm)], np.int32)
            if len(pts_inner) > 0:
                cv2.polylines(frame, [pts_inner], True, (255, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished {video_path.name}")

def main():
    if not DATA_DIR.exists():
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return

    video_files = sorted(list(DATA_DIR.glob("*.webm")))
    if not video_files:
        print(f"No .webm files found in {DATA_DIR}")
        return

    print(f"Found {len(video_files)} videos in {DATA_DIR}")
    for video_file in video_files:
        process_video(video_file)

if __name__ == "__main__":
    main()
