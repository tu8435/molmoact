"""Utils for evaluating policies in LIBERO simulation environments ported from OpenVLA."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution,
        "render_gpu_device_id": 0,  # Explicitly set to 0 to avoid MIG UUID parsing issues
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img

def get_libero_wrist_image(obs, resize_size):
    """Extracts wrist camera image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, checkpoint, task, task_id=None, base_dir=None, **kwargs):
    """
    Saves an MP4 replay of an episode.
    
    Directory structure:
    {base_dir}/{DATE}/{model_name}/{task_type}/{task_id}/{kwargs_dir}/{filename}.mp4
    
    Args:
        rollout_images: List of images to save as video
        idx: Episode index
        success: Whether episode was successful
        task_description: Description of the task
        checkpoint: Model checkpoint path (used to extract model name)
        task: Task suite name (e.g., "libero_spatial")
        task_id: Task ID (0-9)
        base_dir: Base directory for rollouts (default: experiments/libero/rollouts relative to script location)
        **kwargs: Additional parameters to include in path (e.g., noise_level=15.0)
    """
    # Set base directory - use absolute path
    if base_dir is None:
        # Get the directory where this script is located (libero_utils.py is in experiments/libero/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "rollouts")
    
    # Extract model name from checkpoint path
    # Checkpoint can be either a hub path like "allenai/MolmoAct-7B-D-LIBERO-Spatial-0812"
    # or a full path like "/path/to/models--allenai--MolmoAct-7B-D-LIBERO-Spatial-0812/snapshots/..."
    if "/" in checkpoint or "\\" in checkpoint:
        # Extract model name from path
        # Look for "models--" pattern in path
        if "models--" in checkpoint:
            # Extract everything after "models--" until the next "/"
            model_part = checkpoint.split("models--")[-1].split("/")[0].split("\\")[0]
            model_name = f"models--{model_part}"
        else:
            # Try to extract from hub-style path
            parts = checkpoint.replace("\\", "/").split("/")
            if len(parts) >= 2:
                # Assume format like "allenai/MolmoAct-7B-D-LIBERO-Spatial-0812"
                model_name = f"models--{parts[-2]}--{parts[-1]}"
            else:
                # Fallback: use checkpoint name directly
                model_name = checkpoint.replace("/", "--").replace("\\", "--")
    else:
        # Simple checkpoint name, convert to model format
        model_name = f"models--{checkpoint.replace('/', '--')}"
    
    # Extract task type from task suite name (remove "libero_" prefix)
    task_type = task.replace("libero_", "") if task.startswith("libero_") else task
    
    # Build kwargs directory name
    kwargs_parts = []
    if kwargs:
        for key, value in sorted(kwargs.items()):
            # Format value for directory name
            if isinstance(value, bool):
                # Format booleans more cleanly (e.g., include_trace -> "with_trace" or "no_trace")
                if key == "include_trace":
                    # Use cleaner name without the key prefix for include_trace
                    kwargs_parts.append("with_trace" if value else "no_trace")
                    continue
                else:
                    value_str = "yes" if value else "no"
            elif isinstance(value, float):
                value_str = f"{value:.1f}".rstrip('0').rstrip('.')
            else:
                value_str = str(value)
            kwargs_parts.append(f"{key}_{value_str}")
    
    kwargs_dir = "_".join(kwargs_parts) if kwargs_parts else "default"
    
    # Build directory path: base_dir/DATE/model_name/task_type/task_id/kwargs_dir
    if task_id is not None:
        rollout_dir = os.path.join(base_dir, DATE, model_name, task_type, str(task_id), kwargs_dir)
    else:
        rollout_dir = os.path.join(base_dir, DATE, model_name, task_type, kwargs_dir)
    
    os.makedirs(rollout_dir, exist_ok=True)
    
    # Create filename
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = os.path.join(rollout_dir, f"{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4")
    
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den