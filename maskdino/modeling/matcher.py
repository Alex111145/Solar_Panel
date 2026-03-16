# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

# Import corretto con risalita di una cartella
from ..utils import box_ops


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        cost_box: float = 1, # Nome corretto per MaskDINO
        cost_giou: float = 1,
        num_points: int = 0,
    ):
        """Initializes the matcher"""
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box # Deve chiamarsi cost_box
        self.cost_giou = cost_giou
        self.num_points = num_points
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0 or cost_box != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for i in range(bs):
            out_prob = outputs["pred_logits"][i].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[i]["labels"]

            # 1. Classification cost
            cost_class = -out_prob[:, tgt_ids]

            # 2. Box costs
            out_bbox = outputs["pred_boxes"][i]
            tgt_bbox = targets[i]["boxes"]
            
            # L1 cost
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # GIoU cost
            cost_giou = -box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(out_bbox),
                box_ops.box_cxcywh_to_xyxy(tgt_bbox)
            )

            # 3. Final Cost Matrix
            # Usiamo self.cost_box come richiesto dall'errore
            C = (
                self.cost_box * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            
            # --- SAFETY CHECK: PROTECT AGAINST NaN/Inf ---
            C = torch.where(torch.isnan(C), torch.full_like(C, 1e6), C)
            C = torch.where(torch.isinf(C), torch.full_like(C, 1e6), C)

            C_cpu = C.detach().cpu()
            
            if C_cpu.shape[1] == 0:
                indices.append((
                    torch.as_tensor([], dtype=torch.int64), 
                    torch.as_tensor([], dtype=torch.int64)
                ))
            else:
                # Esecuzione del matching sulla CPU
                indices.append(linear_sum_assignment(C_cpu))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching"""
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
            "cost_box: {}".format(self.cost_box),
            "cost_giou: {}".format(self.cost_giou),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)