"""
Adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
added
"""

from functools import partial

import torch
import torch.nn.functional as F
import torchvision
import torchvision.models.detection.roi_heads
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads


class FastRCNNPredictorExtraCapacity(nn.Module):
    """Add capacity in the Fast R-CNN  classification + bounding box regression layers
    (used in ablation study on SBL)
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.cls_score = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, num_classes),
        )

        self.bbox_pred = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, num_classes * 4),
        )

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


# Define custom RoIHeads with class-weighted loss
class HierarchicalRoIHeads(RoIHeads):
    """
    Module for taxonomic hierarchical loss.
    """
    def __init__(
        self,
        *args,
        genus2species,
        family2genus,
        species2genus,
        genus2family,
        loss_weights,
        **kwargs,
    ):
        """
        FasterRCNN head with hierarchical classification loss
        genus2species: dictionary {genus_id:[list of species_ids in genus]}
        family2genus: dictionary {family_id:[list of genera_ids in family]}
        species2genus: dictionary {species_id: corresponding genus_id}
        genus2family: dictionary {genus_id: corresponding family_id}
        loss_weights: triplet of weights for species loss, genus loss, family loss
        """
        super().__init__(*args, **kwargs)
        self.num_genus = len(genus2species)
        self.num_families = len(family2genus)
        self.num_species = len(species2genus)

        self.genus2species = genus2species
        self.family2genus = family2genus
        self.species2genus = torch.IntTensor(
            [species2genus[i] for i in range(self.num_species)]
        )
        self.genus2family = torch.IntTensor(
            [genus2family[i] for i in range(self.num_genus)]
        )

        self.sp_loss = nn.NLLLoss()
        self.gn_loss = nn.NLLLoss()
        self.fm_loss = nn.NLLLoss()
        self.softmax = nn.Softmax(dim=1)

        self.w1, self.w2, self.w3 = loss_weights
        self.epsilon = 1e-6

    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        """Custom Fast R-CNN classification loss with per-class weights."""
        # make sure that there has been a class 0 added to the species, genus and families mapping to 0

        labels = torch.cat(labels, dim=0)

        genus_labels = self.species2genus[labels]
        family_labels = self.genus2family[genus_labels]
        species_logits = self.softmax(class_logits)

        # Create a tensor from index_list
        genus_index_tensor = [
            torch.tensor(indices) for indices in list(self.genus2species.values())
        ]
        genus_preds = torch.stack(
            [species_logits[:, indices].sum(dim=1) for indices in genus_index_tensor],
            dim=1,
        )

        family_index_tensor = [
            torch.tensor(indices) for indices in list(self.family2genus.values())
        ]
        family_preds = torch.stack(
            [genus_preds[:, indices].sum(dim=1) for indices in family_index_tensor],
            dim=1,
        )

        regression_targets = torch.cat(regression_targets, dim=0)

        # add epsilon
        species_loss = self.sp_loss(torch.log(species_logits + self.epsilon), labels)
        genus_loss = self.gn_loss(torch.log(genus_preds + self.epsilon), genus_labels)
        family_loss = self.fm_loss(
            torch.log(family_preds + self.epsilon), family_labels
        )

        classification_loss = (
            self.w1 * species_loss + self.w2 * genus_loss + self.w3 * family_loss
        )

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ResNetEncodeDSMModel(torch.nn.Module):
    """
    Adapt Mask R-CNN to take as input 4th (DSM) channel.
    """
    def __init__(self):
        super(ResNetEncodeDSMModel, self).__init__()
        self.resnet = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406, 0],
            image_std=[0.229, 0.224, 0.225, 1],
        )
        self.resnet.transform = grcnn
        old_weights = self.resnet.backbone.body.conv1.weight.data[:, :3, :, :].clone()
        self.resnet.backbone.body.conv1 = nn.Conv2d(
            4,
            self.resnet.backbone.body.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        mask_in_chans = 768

        self.resnet.backbone.body.conv1.weight.data[:, :3, :, :] = old_weights
        self.backbonedsm = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, padding="same"),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, padding="same"),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, 1, kernel_size=1),
        )

    def forward(self, x, dsm, targets=None):

        dsm = self.backbonedsm(dsm)

        x = torch.cat((x, dsm), dim=1)
        if targets is None:
            x = self.resnet(x)
        else:
            x = self.resnet(x, targets)
        return x


WEIGHTS = torch.Tensor(
    [
        0,
        0.03758783106,
        0.0003242992819,
        0.006007258127,
        0.04879932052,
        0.07990116593,
        0.007427997838,
        0.0005713844491,
        0.0001235425836,
        0.00560574473,
        0.0683036059,
        0.01757393251,
        0.03547216431,
        0.06030422361,
        0.2732298664,
        0.03246081384,
        0.004462975832,
        0.3007335341,
        0.02111033897,
    ]
)
# this is only for the SBL dataset. These weights were computed as the inverse frequncy of class occurrences in the training set 


def fastrcnn_hierarchical_loss(
    class_logits,
    box_regression,
    labels,
    regression_targets,
    genus2species,
    family2species,
    species2genus,
    species2family,
    loss_weights,
    ignore_species=[],
    ignore_genus=[],
):
    """
    Custom Fast R-CNN hierarchical taxonomic classification loss.
    FasterRCNN head with hierarchical classification loss
    
    genus2species: dictionary {genus_id:[list of species_ids in genus]}
    family2genus: dictionary {family_id:[list of genera_ids in family]}
    species2genus: dictionary {species_id: corresponding genus_id}
    genus2family: dictionary {genus_id: corresponding family_id}
    loss_weights: triplet of weights for species loss, genus loss, family loss
    ignore_species: class indices that are only at genus / family level for which we do not want to compute the species loss
    ignore_genus: class indices that are only at family level for which we do not want to compute the genus loss
    """
    # make sure that there has been a class 0 added to the species, genus and families mapping to 0
    

    labels = torch.cat(labels, dim=0)
    num_genus = len(genus2species)
    num_families = len(family2species)
    num_species = len(species2genus)

    species2genus = torch.Tensor([species2genus[i] for i in range(num_species)])
    species2family = torch.Tensor([species2family[i] for i in range(num_species)])
    family_labels = species2family[labels.cpu()].long().to("cuda")
    genus_labels = species2genus[labels.cpu()].long().to("cuda")
    sp_loss = nn.NLLLoss()
    gn_loss = nn.NLLLoss()
    fm_loss = nn.NLLLoss()
    softmax = nn.Softmax(dim=1)

    w1, w2, w3 = loss_weights
    epsilon = 1e-6

    # family_labels = species2family[labels].long().to("cuda") #self.genus2family.unsqueeze(0).expand(genus_labels.shape[0], *self.genus2family.shape)[genus_labels]

    all_species_logits = softmax(class_logits)

    # Create a tensor from index_list

    family_index_tensor = [
        torch.tensor(indices) for indices in list(family2species.values())
    ]
    family_preds = torch.stack(
        [all_species_logits[:, indices].sum(dim=1) for indices in family_index_tensor],
        dim=1,
    ).to("cuda")

    genus_index_tensor = [torch.tensor(ind) for ind in list(genus2species.values())]
    genus_preds = torch.stack(
        [all_species_logits[:, indices].sum(dim=1) for indices in genus_index_tensor],
        dim=1,
    ).to("cuda")

    regression_targets = torch.cat(regression_targets, dim=0)

    if len(ignore_species) > 0:

        # Get indices of elements NOT in exclude_values
        indices = torch.where(~torch.isin(labels, ignore_species))[0]

        species_logits = all_species_logits[indices]
        species_labels = labels[indices]

    if len(ignore_genus) > 0:
        # exclude ground truth instances labelled Magnoliopsida and Pinopsida.

        # Get indices of elements NOT in exclude_values
        indices = torch.where(~torch.isin(labels, ignore_genus))[0]
        genus_preds = genus_preds[indices]
        genus_labels = genus_labels[indices]

    # add epsilon
    species_loss = sp_loss(torch.log(species_logits + epsilon), species_labels)
    genus_loss = gn_loss(torch.log(genus_preds + epsilon), genus_labels)
    family_loss = fm_loss(torch.log(family_preds + epsilon), family_labels)

    classification_loss = w1 * species_loss + w2 * genus_loss + w3 * family_loss

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def fastrcnn_loss_weights(
    class_logits, box_regression, labels, regression_targets, weight=WEIGHTS
):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """Custom Fast R-CNN classification loss with per-class weights.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
        weight: list of weigths for each class.

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    weight = torch.Tensor(weight).to(class_logits.device)
    classification_loss = F.cross_entropy(class_logits, labels, weight=weight)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class NoTransform(nn.Module):
    """A no-op transform that returns images and targets unchanged."""

    def __call__(self, images, targets=None):
        return images, targets  # Return unmodified images and targets

    def postprocess(self, result, image_shapes, original_image_sizes):
        return result  # No changes to postprocessing


def get_model_instance_segmentation(
    num_classes, model_version="resnet50", weights_loss=None, hierarchical_loss=None
):

    if model_version == "resnet50":
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    elif model_version == "resnet50scratch":
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    elif model_version == "resnet50scratchonlydsm":
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800, max_size=1333, image_mean=[0], image_std=[1]
        )  # [337.503], image_std=[ 71.9472])
        model.transform = grcnn
        model.backbone.body.conv1 = nn.Conv2d(
            1,
            model.backbone.body.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
    elif model_version == "resnet50scratchonlydsmplantations":
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800, max_size=1333, image_mean=[0], image_std=[1]
        )
        model.transform = grcnn
        model.backbone.body.conv1 = nn.Conv2d(
            1,
            model.backbone.body.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    elif model_version == "resnet50dsm" or model_version == "resnet50dsm_extracapacity":
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406, 0],
            image_std=[0.229, 0.224, 0.225, 1],
        )
        model.transform = grcnn
        old_weights = model.backbone.body.conv1.weight.data[:, :3, :, :].clone()
        model.backbone.body.conv1 = nn.Conv2d(
            4,
            model.backbone.body.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model.backbone.body.conv1.weight.data[:, :3, :, :] = old_weights

    elif model_version == "resnet50dsmgrads":
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406, 0, 0],
            image_std=[0.229, 0.224, 0.225, 1, 1],
        )
        model.transform = grcnn
        old_weights = model.backbone.body.conv1.weight.data[:, :3, :, :].clone()
        model.backbone.body.conv1 = nn.Conv2d(
            5,
            model.backbone.body.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model.backbone.body.conv1.weight.data[:, :3, :, :] = old_weights
    elif model_version == "resnet50scratchdsm":
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406, 0],
            image_std=[0.229, 0.224, 0.225, 1],
        )

        model.transform = grcnn
        print("image_mean", grcnn.image_mean)
        model.backbone.body.conv1 = nn.Conv2d(
            4,
            model.backbone.body.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    elif model_version == "resnet50_v2":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    elif model_version == "resnet50encodedsm":
        model = ResNetEncodeDSMModel()
        # get number of input features for the classifier
        in_features = model.resnet.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.resnet.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        # now get the number of input features for the mask classifier
        in_features_mask = model.resnet.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.resnet.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        return model

    else:
        raise ValueError("model name for mask R-CNN is not handled.")

    if weights_loss is not None:

        torchvision.models.detection.roi_heads.fastrcnn_loss = partial(
            fastrcnn_loss_weights, weight=weights_loss
        )

    if hierarchical_loss == "quebec_trees":
        ignore_species = [
            2,
            3,
            8,
            11,
        ]  # don't compute species loss for pinopsida and magnoliopsida, Betula and acer
        ignore_genus = [2, 3]
        loss_weights = [1, 0.3, 0.1]
        genus2species = {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [5],
            6: [6],
            7: [7],
            8: [8, 16, 17],
            9: [9],
            10: [10],
            11: [11, 12, 13, 14],
            12: [15],
            13: [18],
        }
        species2genus = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 11,
            13: 11,
            14: 11,
            15: 12,
            16: 8,
            17: 8,
            18: 13,
        }
        family2species = {
            0: [0],
            1: [1],
            2: [2, 4, 5, 6, 7, 15, 18],
            3: [3, 8, 9, 10, 11, 12, 13, 14, 16, 17],
        }
        species2family = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 2,
            5: 2,
            6: 2,
            7: 2,
            8: 3,
            9: 3,
            10: 3,
            11: 3,
            12: 3,
            13: 3,
            14: 3,
            15: 2,
            16: 3,
            17: 3,
            18: 2,
        }
        # family2genus = {0:[0], 1:[1], 2:[2,4,5,6,7,12,13],  3:[3,8,9,10,11] }
        # genus2family = {0:0, 1:1, 2:2, 3:3, 4:2, 5:2, 6:2, 7:2, 8:3, 9:3, 10:3, 11:3, 12:3, 13:3, 14:3, 15:2, 16:3, 17:3, 18:2}
        ignore_species = torch.tensor(ignore_species).to("cuda")
        ignore_genus = torch.tensor(ignore_genus).to("cuda")
        torchvision.models.detection.roi_heads.fastrcnn_loss = partial(
            fastrcnn_hierarchical_loss,
            genus2species=genus2species,
            family2species=family2species,
            species2genus=species2genus,
            species2family=species2family,
            loss_weights=loss_weights,
            ignore_species=ignore_species,
            ignore_genus=ignore_genus,
        )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    if model_version == "resnet50dsm_extracapacity":
        model.roi_heads.box_predictor = FastRCNNPredictorExtraCapacity(
            in_features, num_classes
        )

    return model
