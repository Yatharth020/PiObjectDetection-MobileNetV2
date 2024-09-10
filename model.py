import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image
import os
import json
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class AnchorGenerator:
    def __init__(self, sizes, aspect_ratios):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = self._calculate_anchors()

    def _calculate_anchors(self):
        anchors = []
        for size in self.sizes:
            area = size ** 2
            for aspect_ratio in self.aspect_ratios:
                w = np.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(anchors)

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size in grid_sizes:
            grid_height, grid_width = size
            shifts_x = torch.arange(0, grid_width) / grid_width
            shifts_y = torch.arange(0, grid_height) / grid_height
            shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=2).reshape(-1, 1, 4)
            anchors.append((shifts + self.cell_anchors).reshape(-1, 4))
        return anchors

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

    def forward(self, x):
        results = []
        last_inner = self.inner_blocks[-1](x[-1])
        results.append(self.layer_blocks[-1](last_inner))

        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))

        return results

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, num_anchors=9):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=True).features
        self.fpn = FeaturePyramidNetwork([24, 32, 96, 1280], 256)
        
        self.classification_head = nn.ModuleList([
            nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1)
            for _ in range(5)
        ])
        
        self.regression_head = nn.ModuleList([
            nn.Conv2d(256, num_anchors * 4, kernel_size=3, padding=1)
            for _ in range(5)
        ])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [3, 6, 13, 18]:
                features.append(x)
        
        fpn_features = self.fpn(features)
        
        class_logits = []
        bbox_regression = []
        
        for feature, cls_head, reg_head in zip(fpn_features, self.classification_head, self.regression_head):
            class_logits.append(cls_head(feature).permute(0, 2, 3, 1).contiguous())
            bbox_regression.append(reg_head(feature).permute(0, 2, 3, 1).contiguous())
        
        return class_logits, bbox_regression

def calc_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.annotations = json.load(open(os.path.join(root_dir, 'annotations.json')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.root_dir, img_name)).convert('RGB')
        
        annotation = self.annotations[img_name]
        boxes = torch.tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.tensor(annotation['labels'], dtype=torch.long)
        
        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
        
        return image, {'boxes': boxes, 'labels': labels}

def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return torch.stack(images, 0), targets

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, num_classes):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.focal_loss = FocalLoss()
        self.anchor_generator = AnchorGenerator(sizes=[32, 64, 128, 256, 512],
                                                aspect_ratios=[0.5, 1.0, 2.0])

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for images, targets in tepoch:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                self.optimizer.zero_grad()
                
                class_logits, bbox_regression = self.model(images)
                
                loss = self.compute_loss(class_logits, bbox_regression, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                
                tepoch.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)

    def compute_loss(self, class_logits, bbox_regression, targets):
        device = class_logits[0].device
        
        cls_losses = []
        reg_losses = []
        
        for cls_per_level, reg_per_level in zip(class_logits, bbox_regression):
            anchors = self.anchor_generator.grid_anchors([cls_per_level.shape[1:3]])
            anchors = anchors[0].to(device)
            
            matched_idxs = []
            class_targets = []
            reg_targets = []
            
            for targets_per_image in targets:
                match_quality_matrix = calc_iou(targets_per_image["boxes"], anchors)
                matched_idxs_per_image = match_quality_matrix.max(dim=0)[1]
                
                matched_idxs.append(matched_idxs_per_image)
                class_targets.append(targets_per_image["labels"][matched_idxs_per_image])
                reg_targets.append(self.encode_boxes(anchors, targets_per_image["boxes"][matched_idxs_per_image]))
            
            class_targets = torch.cat(class_targets, dim=0)
            reg_targets = torch.cat(reg_targets, dim=0)
            
            cls_loss = self.focal_loss(cls_per_level.view(-1, self.num_classes), 
                                       F.one_hot(class_targets, num_classes=self.num_classes).float())
            reg_loss = F.smooth_l1_loss(reg_per_level.view(-1, 4), reg_targets)
            
            cls_losses.append(cls_loss)
            reg_losses.append(reg_loss)
        
        return sum(cls_losses) + sum(reg_losses)

    def encode_boxes(self, reference_boxes, proposals):
        wx = 10.0
        wy = 10.0
        ww = 5.0
        wh = 5.0
        
        ex_ctr_x = (proposals[:, 0] + proposals[:, 2]) / 2
        ex_ctr_y = (proposals[:, 1] + proposals[:, 3]) / 2
        ex_widths = proposals[:, 2] - proposals[:, 0]
        ex_heights = proposals[:, 3] - proposals[:, 1]
        
        gt_ctr_x = (reference_boxes[:, 0] + reference_boxes[:, 2]) / 2
        gt_ctr_y = (reference_boxes[:, 1] + reference_boxes[:, 3]) / 2
        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, unit="batch"):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                class_logits, bbox_regression = self.model(images)
                
                loss = self.compute_loss(class_logits, bbox_regression, targets)
                total_loss += loss.item()
                
                predictions = self.post_process_detections(class_logits, bbox_regression)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        mAP = self.calculate_mAP(all_predictions, all_targets)
        return total_loss / len(self.val_loader), mAP

    def post_process_detections(self, class_logits, bbox_regression):
        device = class_logits[0].device
        results = []
        
        for cls_per_level, reg_per_level in zip(class_logits, bbox_regression):
            anchors = self.anchor_generator.grid_anchors([cls_per_level.shape[1:3]])
            anchors = anchors[0].to(device)
            
            class_prob = F.softmax(cls_per_level, dim=-1)
            box_regression = reg_per_level.reshape(cls_per_level.shape[0], -1, 4)
            
            num_classes = class_prob.shape[-1]
            
            for i in range(cls_per_level.shape[0]): 
                class_prob_per_image = class_prob[i]
                box_regression_per_image = box_regression[i]
                
                results_per_image = []
                
                for cls in range(1, num_classes):
                    scores = class_prob_per_image[..., cls].flatten()
                    boxes = self.decode_boxes(anchors, box_regression_per_image)
                    
                    keep = nms(boxes, scores, iou_threshold=0.5)
                    
                    boxes = boxes[keep]
                    scores = scores[keep]
                    
                    results_per_image.append({
                        'boxes': boxes,
                        'scores': scores,
                        'labels': torch.full((len(boxes),), cls, dtype=torch.int64, device=device)
                    })
                
                results.append(results_per_image)
        
        return results

    def decode_boxes(self, reference_boxes, proposals):
        wx, wy, ww, wh = 10.0, 10.0, 5.0, 5.0
        
        widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        ctr_x = reference_boxes[:, 0] + 0.5 * widths
        ctr_y = reference_boxes[:, 1] + 0.5 * heights
        
        dx = proposals[:, 0] / wx
        dy = proposals[:, 1] / wy
        dw = proposals[:, 2] / ww
        dh = proposals[:, 3] / wh
        
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ], dim=1)
        
        return pred_boxes

    def calculate_mAP(self, all_predictions, all_targets):
        aps = []
        for cls in range(1, self.num_classes):
            predictions = [pred[cls - 1] for pred in all_predictions if len(pred) > cls - 1]
            targets = [{'boxes': t['boxes'][t['labels'] == cls], 'labels': t['labels'][t['labels'] == cls]} 
                       for t in all_targets]
            
            ap = self.average_precision_per_class(predictions, targets)
            aps.append(ap)
        
        return sum(aps) / len(aps)

    def average_precision_per_class(self, predictions, targets):
        num_targets = sum(len(t['boxes']) for t in targets)
        
        sorted_indices = torch.cat([p['scores'] for p in predictions]).argsort(descending=True)
        sorted_predictions = torch.cat([p['boxes'] for p in predictions])[sorted_indices]
        sorted_scores = torch.cat([p['scores'] for p in predictions])[sorted_indices]
        
        tp = torch.zeros_like(sorted_scores)
        fp = torch.zeros_like(sorted_scores)
        
        for i, (pred_box, score) in enumerate(zip(sorted_predictions, sorted_scores)):
            best_iou = 0
            best_target_idx = -1
            
            for j, target in enumerate(targets):
                if len(target['boxes']) == 0:
                    continue
                
                ious = calc_iou(pred_box.unsqueeze(0), target['boxes'])
                max_iou, max_idx = ious.max(dim=0)
                
                if max_iou > best_iou:
                    best_iou = max_iou
                    best_target_idx = j
            
            if best_iou > 0.5:
                if not targets[best_target_idx].get('used', torch.zeros_like(targets[best_target_idx]['labels'])).any():
                    tp[i] = 1
                    targets[best_target_idx]['used'] = torch.ones_like(targets[best_target_idx]['labels'])
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumsum / num_targets
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        recalls = torch.cat([torch.tensor([0]), recalls, torch.tensor([1])])
        precisions = torch.cat([torch.tensor([1]), precisions, torch.tensor([0])])
        
        return torch.trapz(precisions, recalls)

    def train(self, num_epochs):
        best_mAP = 0
        train_losses = []
        val_losses = []
        mAPs = []
        
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss, mAP = self.evaluate()
            
            self.scheduler.step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            mAPs.append(mAP)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mAP: {mAP:.4f}')
            
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            if epoch > 10 and val_loss > min(val_losses[:-10]):
                print('Early stopping')
                break
        
        self.plot_metrics(train_losses, val_losses, mAPs)
    
    def plot_metrics(self, train_losses, val_losses, mAPs):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        
        plt.subplot(1, 2, 2)
        plt.plot(mAPs, label='mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.title('Mean Average Precision')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

def get_transforms():
    train_transform = A.Compose([
        A.RandomResizedCrop(300, 300),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    val_transform = A.Compose([
        A.Resize(300, 300),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    return train_transform, val_transform

def create_data_loaders(data_dir, batch_size=32, train_ratio=0.8):
    all_images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    train_images, val_images = train_test_split(all_images, train_size=train_ratio, random_state=42)

    train_transform, val_transform = get_transforms()

    train_dataset = CustomDataset(data_dir, transform=train_transform)
    val_dataset = CustomDataset(data_dir, transform=val_transform)

    train_dataset.images = train_images
    val_dataset.images = val_images

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader

def create_optimizer(model, lr=1e-3, weight_decay=5e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def create_scheduler(optimizer, num_epochs, steps_per_epoch):
    return OneCycleLR(optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

def inference(model, image_path, device, num_classes):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms()[1] 
    image_tensor = transform(image=np.array(image))['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        class_logits, bbox_regression = model(image_tensor)
    
    predictions = model.post_process_detections(class_logits, bbox_regression)[0]
    
    return predictions

