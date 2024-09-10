import io
import time
import picamera
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from threading import Thread, Event
import cv2

from model import ModifiedMobileNetV2, AnchorGenerator, nms

class PiCameraStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.camera = picamera.PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.raw_capture = io.BytesIO()
        self.stream = self.camera.capture_continuous(self.raw_capture, format="rgb", use_video_port=True)
        self.frame = None
        self.stopped = False
        self.event = Event()

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        for f in self.stream:
            self.raw_capture.seek(0)
            self.frame = Image.frombytes("RGB", self.camera.resolution, self.raw_capture.getvalue())
            self.raw_capture.truncate(0)
            self.event.set()
            if self.stopped:
                self.stream.close()
                self.raw_capture.close()
                self.camera.close()
                return

    def read(self):
        self.event.wait()
        self.event.clear()
        return self.frame

    def stop(self):
        self.stopped = True

class ObjectDetector:
    def __init__(self, model_path, num_classes=10, confidence_threshold=0.5, nms_threshold=0.3, device='cpu'):
        self.device = torch.device(device)
        self.model = ModifiedMobileNetV2(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.anchor_generator = AnchorGenerator(sizes=[32, 64, 128, 256, 512],
                                                aspect_ratios=[0.5, 1.0, 2.0])

    def detect(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            class_logits, bbox_regression = self.model(image_tensor)
        
        return self.post_process_detections(class_logits, bbox_regression)

    def post_process_detections(self, class_logits, bbox_regression):
        device = class_logits[0].device
        results = []
        
        for cls_per_level, reg_per_level in zip(class_logits, bbox_regression):
            anchors = self.anchor_generator.grid_anchors([cls_per_level.shape[1:3]])
            anchors = anchors[0].to(device)
            
            class_prob = torch.nn.functional.softmax(cls_per_level, dim=-1)
            box_regression = reg_per_level.reshape(cls_per_level.shape[0], -1, 4)
            
            num_classes = class_prob.shape[-1]
            
            for i in range(cls_per_level.shape[0]): 
                class_prob_per_image = class_prob[i]
                box_regression_per_image = box_regression[i]
                
                results_per_image = []
                
                for cls in range(1, num_classes): 
                    scores = class_prob_per_image[..., cls].flatten()
                    boxes = self.decode_boxes(anchors, box_regression_per_image)
                    
                    keep = nms(boxes, scores, self.nms_threshold)
                    
                    boxes = boxes[keep]
                    scores = scores[keep]
                    
                    mask = scores > self.confidence_threshold
                    boxes = boxes[mask]
                    scores = scores[mask]
                    
                    results_per_image.append({
                        'boxes': boxes,
                        'scores': scores,
                        'labels': torch.full((len(boxes),), cls, dtype=torch.int64, device=device)
                    })
                
                results.append(results_per_image)
        
        return results[0] 
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

def draw_detections(image, predictions, class_names):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for pred in predictions:
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            box = box.cpu().numpy()
            label = label.item()
            score = score.item()
            
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label_text = f"{class_names[label]}: {score:.2f}"
            draw.text((x1, y1 - 10), label_text, fill="red", font=font)
    
    return image

