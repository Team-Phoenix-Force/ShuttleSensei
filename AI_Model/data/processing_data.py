import cv2
import os

def frames_to_video(frames_dir, output_video_path_template, fps, subsequences):
    
    # Loop through each subsequence
    for idx, (start, end) in enumerate(subsequences, start=1):
        output_video_path = output_video_path_template.format(idx)
        
        # Define video codec and first frame properties
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        first_frame_path = os.path.join(frames_dir, f"frame_{start:06d}.png")
        frame = cv2.imread(first_frame_path)
        height, width, _ = frame.shape
        
        # Create video writer object
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Write frames to video
        for i in range(start, end + 1):
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
            frame = cv2.imread(frame_path)
            out.write(frame)
        out.release()
    cv2.destroyAllWindows()

def split_video_into_frames(video_path, output_folder):

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open video capture object
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Loop through each frame, extract, and save
    for frame_number in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:06d}.png")
        cv2.imwrite(frame_filename, frame)
    
    # Release video capture object
    cap.release()
