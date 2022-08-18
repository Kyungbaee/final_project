from unicodedata import category
import streamlit as st

st.set_page_config(
    page_title="Seoul_Landmark",
    page_icon="🗽",
    layout="wide",
)

import torch
import torchvision
from torchvision import models
import torch.nn as nn

from torch.utils.data import DataLoader 
from torch.utils.data import DataLoader, Dataset
import albumentations
import albumentations.pytorch
import cv2
from PIL import Image

import numpy


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2018/07/05/16/59/students-3518726_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.sidebar.markdown("# Seoul_Landmark Dataset ")

st.sidebar.markdown("# Page 2")

st.markdown("### Page 2 이미지 분류 ")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model():
    
    model = models.resnet50(pretrained=True).to(device)
    model.fc = nn.Linear(model.fc.in_features, 3).to(device)
    # state_dict = torch.utils.model_zoo.load_url('https://drive.google.com/uc?export=download&id=1-3SvCFcqdaecIZzziq6PnfFSKFj7n3Os&confirm=t')
    state_dict = torch.load('weights/best_model_weight_8_17.pt',map_location=torch.device('cpu'))
    
    # state_dict = torch.load(r'C:\Users\pc\Desktop\streamlit\final_project\final_project\weights\best_model_weight_8_17.pt')

    model.load_state_dict(state_dict['net'])
    return model


albumentations_test = albumentations.Compose([
                                                
    albumentations.Resize(128, 128, always_apply=True),
    albumentations.Normalize(),
    albumentations.pytorch.transforms.ToTensorV2()
    
])

class CustomDataset(Dataset):
    def __init__(self, img_path, label, train_mode=True, transforms=None): 
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path = img_path
        self.label = label
        

    def __getitem__(self, index): 
        img_path = self.img_path
        image = img_path

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented['image'] 

        if self.train_mode:
            label = self.label[index]
            return image, label
        else:
            return image
    
    def __len__(self): 
        # return len(self.img_path)
        return 1


def test(model, vali_loader):
    model.eval()
    with torch.no_grad():

        for k, data in enumerate(vali_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
        return predicted.item()


@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img



image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

	st.image(load_image(image_file),width=250)

if image_file is not None:
    I = numpy.asarray(Image.open(image_file).convert("RGB"))

    model = load_model()
    vali_dataset = CustomDataset(I,[0], train_mode=True, transforms=albumentations_test)
    vali_loader = DataLoader(vali_dataset, batch_size = 16, shuffle=False, num_workers=0)

    theme = test(model, vali_loader)

    if theme == 0:
        st.write("")
        st.write("## 🏢 전시 입니다.")
    elif theme == 1:
        st.write("")
        st.write("## 🏢 야외 입니다.")
    elif theme == 2:
        st.write("")
        st.write("## 🏢 체험 입니다.")
    

    
