import torch

def softnms_v2(segments, sigma=0.5, top_k=1000, score_threshold=0.001):
    segments = segments.cpu()
    tstart = segments[:, 0]
    tend = segments[:, 1]
    tscore = segments[:, 2]
    done_mask = tscore < -1  # set all to False
    undone_mask = tscore >= score_threshold
    while undone_mask.sum() > 1 and done_mask.sum() < top_k:
        idx = tscore[undone_mask].argmax()
        idx = undone_mask.nonzero()[idx].item()

        undone_mask[idx] = False
        done_mask[idx] = True

        top_start = tstart[idx]
        top_end = tend[idx]
        _tstart = tstart[undone_mask]
        _tend = tend[undone_mask]
        tt1 = _tstart.clamp(min=top_start)
        tt2 = _tend.clamp(max=top_end)
        intersection = torch.clamp(tt2 - tt1, min=0)
        duration = _tend - _tstart
        tmp_width = torch.clamp(top_end - top_start, min=1e-5)
        iou = intersection / (tmp_width + duration - intersection)
        scales = torch.exp(-iou ** 2 / sigma)
        tscore[undone_mask] *= scales
        undone_mask[tscore < score_threshold] = False
    count = done_mask.sum()
    segments = torch.stack([tstart[done_mask], tend[done_mask], tscore[done_mask]], -1)
    return segments, count