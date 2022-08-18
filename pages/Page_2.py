import streamlit as st
# from tkinter.tix import COLUMN
from pyparsing import empty
import torch
import torchvision
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.utils.data import DataLoader, Dataset
import albumentations
import albumentations.pytorch
from PIL import Image
import numpy

st.set_page_config(
    page_title="Seoul_Landmark",
    page_icon="🗽",
    layout="wide",
)
empty1,con1 = st.columns([0.3,1.0])
empty1,con2,con3 = st.columns([0.3,0.5,0.5])

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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model():
    
    model = models.resnet50(pretrained=True).to(device)
    model.fc = nn.Linear(model.fc.in_features, 3).to(device)
    state_dict = torch.utils.model_zoo.load_url('https://drive.google.com/uc?export=download&id=1-3SvCFcqdaecIZzziq6PnfFSKFj7n3Os&confirm=t',map_location=torch.device('cpu'))
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

def main() :

    with empty1 :
        empty() # 여백부분1
        st.sidebar.markdown("# Seoul_Landmark Dataset ")
        st.sidebar.markdown("# Page 2")
        
    with con1 :
        st.markdown("### Page 2 이미지 분류 ")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    
    with con2 :
        if image_file is not None: 
            st.image(load_image(image_file),width=250)

    with con3 :
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
        
main()
            
    






    

    
