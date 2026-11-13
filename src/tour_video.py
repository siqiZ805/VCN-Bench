"""Shared tour-video sampling for the navigation evaluation entrypoints."""

from operator import index

SAMPLING_PROTOCOL = "uniform_endpoints_v1"


def sample_tour_video(frames, max_frames):
    max_frames = index(max_frames)
    if max_frames < 1:
        raise ValueError("video_max_frame must be a positive integer")
    video_len = len(frames)
    sample_count = min(video_len, max_frames)
    sample_idxs = [
        i * (video_len - 1) // max(sample_count - 1, 1) for i in range(sample_count)
    ]
    return [frames[i] for i in sample_idxs]
