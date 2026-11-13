"""Pool MV-DualVLN image features to the grid used by its input tokens."""


def pool_image_features(features, image_grid_thw, target_grid_thw, merge_size):
    import torch
    from torch.nn.functional import adaptive_avg_pool2d

    if target_grid_thw is None:
        return features
    if image_grid_thw.shape != target_grid_thw.shape:
        raise ValueError("Source and target image grids must have matching shapes")
    pooled = []
    offset = 0
    for source, target in zip(image_grid_thw.tolist(), target_grid_thw.tolist()):
        t, raw_h, raw_w = source
        target_t, target_raw_h, target_raw_w = target
        if t != target_t or min(*source, *target) <= 0:
            raise ValueError("Pooling requires positive grids with equal temporal size")
        if any(
            size % merge_size for size in (raw_h, raw_w, target_raw_h, target_raw_w)
        ):
            raise ValueError(
                "Image grid dimensions must be divisible by spatial_merge_size"
            )
        h, w = raw_h // merge_size, raw_w // merge_size
        target_h, target_w = target_raw_h // merge_size, target_raw_w // merge_size
        if target_h > h or target_w > w:
            raise ValueError("The target image grid cannot exceed the source grid")
        count = t * h * w
        image_features = features[offset : offset + count]
        if image_features.shape[0] != count:
            raise ValueError("Image feature count does not match image_grid_thw")
        offset += count
        if (h, w) != (target_h, target_w):
            channels = image_features.shape[-1]
            grid = image_features.reshape(t, h, w, channels).permute(0, 3, 1, 2)
            grid = adaptive_avg_pool2d(grid, (target_h, target_w))
            image_features = grid.permute(0, 2, 3, 1).reshape(-1, channels)
        pooled.append(image_features)
    if offset != features.shape[0]:
        raise ValueError("Unused image features remain after splitting image_grid_thw")
    return torch.cat(pooled, dim=0) if pooled else features
