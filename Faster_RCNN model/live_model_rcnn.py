import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 

from numba import cuda

import os
import xml.etree.ElementTree as ET


from PIL import Image


device = cuda.get_current_device()
print(device)



def predict(frame):
    transform = T.Compose([
        T.ToTensor(),
    ])

    LABEL_MAP = {"full-faced": 1,"half-faced": 2,  "invalid": 3,"no helmet":4}
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, annotation_dir, transforms=None):
            self.root_dir = root_dir
            self.annotation_dir = annotation_dir
            self.transforms = transforms

            # List all XML annotation files
            self.image_files = []
            self.annotation_files = []

            for file in os.listdir(self.annotation_dir):
                if file.endswith(".xml"):
                    self.annotation_files.append(os.path.join(self.annotation_dir, file))
                    self.image_files.append(os.path.join(self.root_dir, file.replace(".xml", ".jpg"))) 

        def parse_annotation(self, xml_file):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            boxes = []
            labels = []

            for obj in root.findall("object"):
                label = obj.find("name").text
                if label in LABEL_MAP:
                    labels.append(LABEL_MAP[label])

                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])

            return boxes, labels

        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            xml_path = self.annotation_files[idx]

            # Load image
            img = Image.open(img_path).convert("RGB")

            # Load annotations
            boxes, labels = self.parse_annotation(xml_path)

            # Convert to PyTorch tensors
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }

            # Apply transforms if specified
            if self.transforms is not None:
                img = self.transforms(img)

            return img, target

        def __len__(self):
            return len(self.image_files)


        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        img = frame
 
        img_tensor = transform(img)
        img_tensor = img_tensor.to(device)

        model = fasterrcnn_resnet50_fpn(pretrained=True,)

        num_classes = 5  

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


        state_dict = torch.load('./faster_rcnn.pt') #gpu

        model.load_state_dict(state_dict) 

        model.eval() 



        img = transform(img)

        img = img.to(device) # pindah gambar ke gpu
        model.to(device)

        with torch.no_grad():
            prediction = model([img]) 
        
        confidence_threshold = 0.5

        filtered_prediction = [{
            'boxes': prediction[0]['boxes'][prediction[0]['scores'] > confidence_threshold],
            'labels': prediction[0]['labels'][prediction[0]['scores'] > confidence_threshold],
            'scores': prediction[0]['scores'][prediction[0]['scores'] > confidence_threshold]
        }]

        print(filtered_prediction[0]['boxes'])
        print(filtered_prediction[0]['labels'])
        print(filtered_prediction[0]['scores'])  

        drawBox(filtered_prediction, frame, confidence_threshold) 

def drawBox(prediction, image, confidence_threshold=0.5):
    try:
        for i in range(len(prediction[0]['boxes'])):
            if 'scores' in prediction[0] and prediction[0]['scores'][i] < confidence_threshold:
                continue
                
            x, y, w, h = prediction[0]['boxes'][i].cpu().numpy().astype('int')
            classes = {1:"full-faced", 2:"half-faced", 3:"invalid", 4:"no helmet"}
            text = classes[prediction[0]['labels'][i].item()]
            
            if 'scores' in prediction[0]:
                text += f" {prediction[0]['scores'][i].item():.2f}"
            
            if text.startswith("full-faced") or text.startswith("half-faced"):
                color = (0, 255, 0)
            elif text.startswith("invalid"):
                color = (255, 255, 0)
            elif text.startswith("no helmet"):
                color = (0, 0, 255)
                
            cv2.rectangle(image, (x, y), (w, h), color, 4)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_w, text_h = text_size
            cv2.rectangle(image, (x, y-text_h), (x + text_w, y + text_h-17), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    except Exception as e:
        print(e)
        print("got no detection")

img = cv2.imread('./image_pres/test_siang.png')


predict(img)
cv2.imwrite(f"C:\\Users\\Michael\\Documents\\python projects\\Karya tulis Computer Vision BHK 2025\\demonstration\\image_pres\\prediction\\faster_cnn_siang.png",img)
# cv2.imshow("liveview",img)
# cv2.imwrite()
cv2.waitKey(0)
