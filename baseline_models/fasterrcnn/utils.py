import datetime
import errno
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)"
        )


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def mask_iou(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> torch.Tensor:
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """

    N, H, W = mask1.shape
    M, H, W = mask2.shape

    mask1 = mask1.view(N, H * W)
    mask2 = mask2.view(M, H * W)

    intersection = torch.matmul(mask1, mask2.t())

    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0.0, device=mask1.device),
        intersection / union,
    )

    return ret


def get_ious(predicted_masks, gt_masks, device, positive_threshold=0.5):
    """
    Args:
        predicted_masks
        gt_masks
        positive_threshold = in the prediction, threshold to consider the prediction is positive
        match_iou = minimum iou to consider there is a match
    Returns:
        matrix of IoUs between predicted_masks and gt_masks

    """
    num_preds = predicted_masks.shape[0]
    num_gt = gt_masks.shape[0]

    ious = torch.zeros((num_preds, num_gt))

    for i in range(num_preds):
        for j in range(num_gt):
            ious[i, j] = mask_iou(
                (predicted_masks[i] > positive_threshold).type(torch.float).to(device),
                gt_masks[j].type(torch.float).to(device),
            )

    return ious


def get_matches(ious, match_threshold=0.2):
    """
    Args:
        ious : matrix of ious (num_preds, num_gt)
        positive_threshold = in the prediction, threshold to consider the prediction is positive
        match_iou = minimum iou to consider there is a match
    Returns:
        vector of size num_gt with the index of the matched predicted mask, or -1 if no match

    """
    ious_match = torch.max(ious, axis=0)
    matches = ious_match.indices
    values = ious_match.values
    matches[values < match_threshold] = -1
    return matches


def compute_metrics_torch(predicted_mask, ground_truth_mask):

    # Basic metrics
    TP = (predicted_mask & ground_truth_mask).sum()
    FP = (predicted_mask & ~ground_truth_mask).sum()
    TN = (~predicted_mask & ~ground_truth_mask).sum()
    FN = (~predicted_mask & ground_truth_mask).sum()

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision and Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # F1 Score
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # Intersection
    intersection = torch.logical_and(predicted_mask, ground_truth_mask).sum()

    # Union
    union = torch.logical_or(predicted_mask, ground_truth_mask).sum()

    # IoU
    iou = intersection / union

    return accuracy, f1_score, iou


def compute_metrics(
    predicted_mask: np.array, ground_truth_mask: np.array
) -> tuple[float, float, float]:
    """
    Compute accuracy, F1 score for binary masks.

    Args:
    - predicted_mask: Predicted binary mask.
    - ground_truth_mask: Ground truth binary mask.

    Returns:
    - accuracy, f1_score, IoU
    """

    # Basic metrics
    TP = np.sum((predicted_mask == 1) & (ground_truth_mask == 1))
    FP = np.sum((predicted_mask == 1) & (ground_truth_mask == 0))
    TN = np.sum((predicted_mask == 0) & (ground_truth_mask == 0))
    FN = np.sum((predicted_mask == 0) & (ground_truth_mask == 1))

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision and Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # F1 Score
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # Intersection
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()

    # Union
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()

    # IoU
    iou = intersection / union

    return accuracy, f1_score, iou


def get_categories(dataset_name):
    if dataset_name == "quebec_trees":
        categories = [
            {
                "id": 1,
                "name": "Dead",
                "global_id": -1,
                "other_names": ["Mort"],
                "supercategory": None,
            },
            {
                "id": 2,
                "name": "Acer",
                "global_id": 3189834,
                "rank": "GENUS",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 3,
                "name": "Picea",
                "global_id": 5284711,
                "rank": "GENUS",
                "other_names": ["PIGL", "PIMA", "PIRU"],
                "supercategory": None,
            },
            {
                "id": 4,
                "name": "Populus",
                "global_id": 3040183,
                "rank": "GENUS",
                "other_names": ["POGR", "POTR"],
                "supercategory": None,
            },
            {
                "id": 5,
                "name": "Betula papyrifera",
                "global_id": 5332120,
                "rank": "SPECIES",
                "other_names": ["BEPA"],
                "supercategory": None,
            },
            {
                "id": 6,
                "name": "Betula alleghaniensis",
                "global_id": 5331779,
                "rank": "SPECIES",
                "other_names": ["BEAL"],
                "supercategory": None,
            },
            {
                "id": 7,
                "name": "Acer saccharum",
                "global_id": 3189859,
                "rank": "SPECIES",
                "other_names": ["ACSA"],
                "supercategory": 2,
            },
            {
                "id": 8,
                "name": "Acer rubrum",
                "global_id": 3189883,
                "rank": "SPECIES",
                "other_names": ["ACRU"],
                "supercategory": 2,
            },
            {
                "id": 9,
                "name": "Acer pensylvanicum",
                "global_id": 179008759,
                "rank": "SPECIES",
                "other_names": ["ACPE"],
                "supercategory": 2,
            },
            {
                "id": 10,
                "name": "Abies balsamea",
                "global_id": 10479128,
                "rank": "SPECIES",
                "other_names": ["ABBA"],
                "supercategory": None,
            },
            {
                "id": 11,
                "name": "Tsuga canadensis",
                "global_id": 2687182,
                "rank": "SPECIES",
                "other_names": ["TSCA"],
                "supercategory": None,
            },
            {
                "id": 12,
                "name": "Pinus strobus",
                "global_id": 5284982,
                "rank": "SPECIES",
                "other_names": ["PIST"],
                "supercategory": None,
            },
            {
                "id": 13,
                "name": "Larix laricina",
                "global_id": 2686231,
                "rank": "SPECIES",
                "other_names": ["LALA"],
                "supercategory": None,
            },
            {
                "id": 14,
                "name": "Thuja occidentalis",
                "global_id": 2684178,
                "rank": "SPECIES",
                "other_names": ["THOC"],
                "supercategory": None,
            },
            {
                "id": 15,
                "name": "Fagus grandifolia",
                "global_id": 2684178,
                "rank": "SPECIES",
                "other_names": ["FAGR"],
                "supercategory": None,
            },
            {
                "id": 16,
                "name": "Betula",
                "global_id": 2875008,
                "rank": "GENUS",
                "other_names": ["BEPO"],
                "supercategory": 2,
            },
            {
                "id": 17,
                "name": "Conifere",
                "global_id": -1,
                "rank": "GROUP",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 18,
                "name": "Feuillus",
                "rank": "GROUP",
                "global_id": -1,
                "other_names": ["FRNI" "OSVI", "POBA", "PRPE", "QURU"],
                "supercategory": None,
            },
        ]

    elif dataset_name == "quebec_plantations":
        categories = [
            {"id": 1, "name": "piba", "supercategory": "pinus"},
            {"id": 2, "name": "pima", "supercategory": "picea"},
            {"id": 3, "name": "pist", "supercategory": "pinus"},
            {"id": 4, "name": "pigl", "supercategory": "picea"},
            {"id": 5, "name": "thoc", "supercategory": "thuja"},
            {"id": 6, "name": "ulam", "supercategory": "ulmaceae"},
            {
                "id": 7,
                "name": "other",
                "other_names": ["qeru", "bepa", "potr", "lala", "pire"],
                "supercategory": "",
            },
            {"id": 8, "name": "beal", "supercategory": "betula"},
            {"id": 9, "name": "acsa", "supercategory": "acer"},
        ]
    elif dataset_name == "bci":
        categories = [
            {
                "id": 1,
                "name": "Burseraceae",
                "global_id": 6659,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 2,
                "name": "Fabaceae",
                "global_id": 5386,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 3,
                "name": "Arecaceae",
                "global_id": 7681,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 4,
                "name": "Rutaceae",
                "global_id": 2396,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 5,
                "name": "Malvaceae",
                "global_id": 6685,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 6,
                "name": "Euphorbiaceae",
                "global_id": 4691,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 7,
                "name": "Bignoniaceae",
                "global_id": 6655,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 8,
                "name": "Annonaceae",
                "global_id": 9291,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 10,
                "name": "Urticaceae",
                "global_id": 6639,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 11,
                "name": "Rubiaceae",
                "global_id": 8798,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 12,
                "name": "Myristicaceae",
                "global_id": 2439,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 13,
                "name": "Apocynaceae",
                "global_id": 6701,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 14,
                "name": "Cordiaceae",
                "global_id": 4930453,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 15,
                "name": "Meliaceae",
                "global_id": 2397,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 16,
                "name": "Other",
                "global_id": None,
                "rank": None,
                "other_names": [
                    "Clusiaceae",
                    "Polygonaceae",
                    "Malpighiaceae",
                    "Myrtaceae",
                    "Erythropalaceae",
                    "Vochysiaceae",
                    "Erythroxylaceae",
                    "Sapindaceae",
                    "Staphyleaceae",
                    "Lythraceae",
                    "Elaeocarpaceae",
                    "Rhizophoraceae",
                    "Monimiaceae",
                    "Violaceae",
                    "Solanaceae",
                ],
                "supercategory": None,
            },
            {
                "id": 17,
                "name": "Simaroubaceae",
                "global_id": 2395,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 18,
                "name": "Moraceae",
                "global_id": 6640,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 19,
                "name": "Anacardiaceae",
                "global_id": 2398,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 20,
                "name": "Sapotaceae",
                "global_id": 8802,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 22,
                "name": "Nyctaginaceae",
                "global_id": 6718,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 23,
                "name": "Lauraceae",
                "global_id": 6688,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 24,
                "name": "Lecythidaceae",
                "global_id": 3990,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 25,
                "name": "Phyllanthaceae",
                "global_id": 8807,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 27,
                "name": "Araliaceae",
                "global_id": 8800,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 29,
                "name": "Salicaceae",
                "global_id": 6664,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 30,
                "name": "Chrysobalanaceae",
                "global_id": 9111,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 31,
                "name": "Cannabaceae",
                "global_id": 2384,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 32,
                "name": "Putranjivaceae",
                "global_id": 8808,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 33,
                "name": "Combretaceae",
                "global_id": 2431,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 39,
                "name": "Melastomataceae",
                "global_id": 6683,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
            {
                "id": 43,
                "name": "Calophyllaceae",
                "global_id": 4907584,
                "rank": "FAMILY",
                "other_names": [],
                "supercategory": None,
            },
        ]
    else:
        raise ValueError("Dataset not supported")
    return categories
