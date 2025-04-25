
import os
import h5py
import glob
import numpy as np
import cv2
origin = "/home/tars/Datasets/piper_joint_rlds/train/*"
episodes = sorted(glob.glob(origin))
print(f"Found {len(episodes)} episodes.")
# Select the first episode for demonstration
episode = episodes[0]


def get_nonstatic_frame_indices(actions, threshold=0.3, num_keep=15):

    # Calculate absolute differences between consecutive frames
    diffs = np.abs(np.diff(actions, axis=0))
    
    # Find frames where any joint's change >= threshold
    non_static_mask = np.any(diffs >= threshold, axis=1)
    
    # Get indices where non-static occurs
    non_static_indices = np.where(non_static_mask)[0]
    
    if len(non_static_indices) == 0:
        return (-1, -1)  # All frames are static
    
    # First non-static frame is the first True in mask (but remember mask is n-1 length)
    first_idx = max(non_static_indices[0] - num_keep, 0)  # +1 because diff compares to previous frame
    
    # Last non-static frame is the last True in mask
    last_idx = min(non_static_indices[-1] + num_keep, len(actions)-1)

    return (first_idx, last_idx)

def save_images_as_mp4(frames, output_path='output.mp4', fps=30):
    """
    Save a sequence of images as an MP4 video file.
    
    Args:
        frames: numpy array of shape (num_frames, height, width, channels) in BGR format
        output_path: path to save the video file (default: 'output.mp4')
        fps: frames per second (default: 30)
    """
    # Get video dimensions
    num_frames, height, width, _ = frames.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each frame to the video
    for i in range(num_frames):
        video_writer.write(frames[i])
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved successfully to {output_path}")

with h5py.File(episode, "r") as F:
    actions = F['frames']['action'][()]
    first_idx, last_idx = get_nonstatic_frame_indices(actions, 0.3)
    states = F['frames']['state'][()]
    images = F['frames']["observation_images_top"][()]
    wrist_images = F['frames']["observation_images_wrist"][()]
    side_images = F['frames']["observation_images_side"][()]
    actions = actions[first_idx:last_idx]
    states = states[first_idx:last_idx]
    images = images[first_idx:last_idx]
    wrist_images = wrist_images[first_idx:last_idx]
    side_images = side_images[first_idx:last_idx]
    # Convert images to BGR format for OpenCV
    images = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images])
    wrist_images = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in wrist_images])
    side_images = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in side_images])
    # Save the images as MP4
    save_images_as_mp4(images, "output.mp4")
    save_images_as_mp4(wrist_images, "output_wrist.mp4")
    save_images_as_mp4(side_images, "output_side.mp4")