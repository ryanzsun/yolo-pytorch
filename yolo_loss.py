import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_dim):
        super(YoloLoss, self).__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss



    def forward(self, inputs, targets):
        n_anchors = self.num_anchors
        n_batches = inputs.size(0)
        n_grids = inputs.size(2)
        stride = self.image_dim / n_grids

        prediction = inputs.view(n_batches, n_anchors, self.bbox_attrs, n_grids, n_grids).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        
        # Calculate offsets for each grid
        grid_x = torch.arange(n_grids).repeat(n_grids, 1).view([1, 1, n_grids, n_grids]).type(FloatTensor)
        grid_y = torch.arange(n_grids).repeat(n_grids, 1).t().view([1, 1, n_grids, n_grids]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, n_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, n_anchors, 1, 1))

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets()

        loss_x = self.mse_loss(x, tx)
        loss_y = self.mse_loss(y, ty)
        loss_w = self.mse_loss(w, tw)
        loss_h = self.mse_loss(h, th)
        loss_conf = self.bce_loss(pred_conf[gt], tconf[gt]) + self.bce_loss(pred_conf[no_gt], tconf[no_gt])
        loss_cls = (1 / n_batches) * self.ce_loss(pred_cls, torch.argmax(tcls, 1))
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return loss