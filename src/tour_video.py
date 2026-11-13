"""Shared tour-video sampling for the navigation evaluation entrypoints."""

SAMPLING_PROTOCOL = "uniform_endpoints_v1"


def sample_tour_video(frames, max_frames):
    """Keep short videos intact; sample longer videos evenly with both endpoints.

    Matches run_memo_e2e_evaluation.py, using the actual sliced video length.
    """
    if max_frames < 1:
        raise ValueError("video_max_frame must be a positive integer")
    video_len = len(frames)
    sample_count = min(video_len, max_frames)
    sample_idxs = [
        i * (video_len - 1) // max(sample_count - 1, 1)
        for i in range(sample_count)
    ]
    return [frames[i] for i in sample_idxs]
