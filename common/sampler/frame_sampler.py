import os
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

class FrameSampler:
    def __init__(
        self,
        max_num_frames: int = 32,
        resolution: Optional[List[int]] = None,
        save_frame: bool = False,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the FrameSampler.
        """
        self.max_num_frames = max_num_frames
        self.resolution = resolution if resolution is not None else []
        self.save_frame = save_frame
        self.output_dir = output_dir

    def save_frames(
        self,
        frames: np.ndarray,
        frame_indices: List[int],
        video_path: str
    ):
        """
        Save sampled frames as images.
        """
        output_dir = self.output_dir or os.path.splitext(video_path)[0] + "_frames"
        os.makedirs(output_dir, exist_ok=True)
        for i, idx in enumerate(frame_indices):
            img = Image.fromarray(frames[i])
            img.save(os.path.join(output_dir, f"frame_{idx}.jpg"))
            
    @staticmethod
    def uniform_sample(frame_list: List[int], num_frames: int) -> List[int]:
        """
        Uniformly sample num_frames indices from frame_list.
        """
        if not num_frames:
            return []
        if num_frames <= 0 or not frame_list:
            return []
        gap = len(frame_list) / num_frames
        idxs = [int(i * gap + gap / 2) for i in range(num_frames)]
        return [frame_list[i] for i in idxs]

    @staticmethod
    def remap_detected_objects_frames(
        detected_objects: List[Dict[str, Any]],
        frame_idx_map: Dict[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Remap the 'frame' field in detected_objects to new indices.
        """
        reindexed = []
        for detection in detected_objects:
            orig_frame = detection.get('frame')
            if orig_frame in frame_idx_map:
                new_detection = detection.copy()
                new_detection['frame'] = frame_idx_map[orig_frame]
                reindexed.append(new_detection)
        return reindexed

    def sample_frames_from_video(
        self,
        video_path: str,
        detected_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Uniformly sample frames from a video, prioritizing detected frames.
        """
        # Load video
        vr_args = {'ctx': cpu(0)}
        if self.resolution:
            vr_args.update({'width': self.resolution[0], 'height': self.resolution[1]})
        vr = VideoReader(video_path, **vr_args)

        # Get frame indices for sampling
        all_frames_idx = list(range(len(vr)))
        detected_frames = [
            d.get('frame') for d in detected_objects if len(d.get('objects', [])) > 0
        ]        
        if len(detected_frames) >= self.max_num_frames:
            frame_idx = self.uniform_sample(detected_frames, self.max_num_frames)
        else:
            remaining_needed = self.max_num_frames - len(detected_frames)
            remaining_frames = [f for f in all_frames_idx if f not in detected_frames]
            sampled_remaining = self.uniform_sample(remaining_frames, remaining_needed) if remaining_needed > 0 else []
            frame_idx = sorted(detected_frames + sampled_remaining)
        
        # Get sampled frames and create a dictionary with reindexed detected objects
        frame_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(frame_idx)}
        reindexed_detected_objects = self.remap_detected_objects_frames(detected_objects, frame_idx_map)
        frames = vr.get_batch(frame_idx).asnumpy()

        # Save frames if required
        if self.save_frame:
            self.save_frames(frames, frame_idx, video_path)

        return {
            "frames": frames,
            "frame_ids": frame_idx,
            "detected_objects": reindexed_detected_objects
        }
