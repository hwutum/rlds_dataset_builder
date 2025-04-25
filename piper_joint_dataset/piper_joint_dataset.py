from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import sys
sys.path.append('.')
from aloha1_put_X_into_pot_300_demos.conversion_utils import MultiThreadedDatasetBuilder

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

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(episode_path):
        # Load raw data
        with h5py.File(episode_path, "r") as F:
            actions = F['frames']['action'][()]
            states = F['frames']['state'][()]
            top_images = F['frames']["observation_images_top"][()]
            side_images = F['frames']["observation_images_side"][()]
            wrist_images = F['frames']["observation_images_wrist"][()]
            # command = F['frames']["language_instruction"][()]
        first_idx, last_idx = get_nonstatic_frame_indices(actions, 0.3, 20)
        num_frames = len(actions)
        actions = actions[first_idx:last_idx][..., :7]
        states = states[first_idx:last_idx][..., :7]
        top_images = top_images[first_idx:last_idx]
        side_images = side_images[first_idx:last_idx]
        wrist_images = wrist_images[first_idx:last_idx]
        print(f"first {first_idx} last {last_idx} remain: {(last_idx-first_idx)/num_frames*100} %")
        command = "r"
        # Assemble episode: here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'top_image': top_images[i],
                    'side_image': side_images[i],
                    'wrist_image': wrist_images[i],
                    'state': np.asarray(states[i], np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
            })

        # Create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # If you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # For smallish datasets, use single-thread parsing
    for sample in paths:
        ret = _parse_example(sample)
        yield ret


class piper_joint_dataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'top_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Top-down camera RGB observation.',
                        ),
                        'side_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Side camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint state (7D left arm + 7D right arm).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot arm action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        datasets = ["piper_joint_rlds"]
        datasets = sum([glob.glob(f"/home/tars/Datasets/{name}/train/episode*.h5") for name in datasets], [])
        
        return {
            "train": datasets
            # "val": datasets,
        }
