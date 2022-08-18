import streamlit as st
st.set_page_config(layout="wide")
import torch
import torchvision
from torchvision import models
import torch.nn as nn

from torch.utils.data import DataLoader 
from torch.utils.data import DataLoader, Dataset
import albumentations
import albumentations.pytorch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model():
    
    model = models.resnet50(pretrained=True).to(device)
    model.fc = nn.Linear(model.fc.in_features, 3).to(device)
    state_dict = torch.load(r'C:\Users\pc\Desktop\streamlit\final_project\weights\best_model_weight_8_17.pt')

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
        index = 0


        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 


        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented['image'] 

        if self.train_mode:
            label = self.label[index]
            return image, label
        else:
            return image
    
    def __len__(self): 
        return len(self.img_path)


def test(model, vali_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        acc = []

        for k, data in enumerate(vali_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc.append(100 * correct/total)

        return 100 * correct/total            




@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img


st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
    img = Image.open(image_file)
    img.save(r"C:\Users\pc\Desktop\streamlit\final_project\img.jpg","png")

if image_file is not None:

	st.image(load_image(image_file),width=250)
    



st.title("이미지 분류")
PATH = r"C:\Users\pc\Desktop\streamlit\final_project\img.jpg"
if st.button('분석하기'):
    model = load_model()
    vali_dataset = CustomDataset(PATH,[0], train_mode=True, transforms=albumentations_test)
    vali_loader = DataLoader(vali_dataset, batch_size = 16, shuffle=False, num_workers=0)
    score0 = test(model, vali_loader)
    vali_dataset = CustomDataset(PATH,[1], train_mode=True, transforms=albumentations_test)
    vali_loader = DataLoader(vali_dataset, batch_size = 16, shuffle=False, num_workers=0)
    score1 = test(model, vali_loader)
    vali_dataset = CustomDataset(PATH,[2], train_mode=True, transforms=albumentations_test)
    vali_loader = DataLoader(vali_dataset, batch_size = 16, shuffle=False, num_workers=0)
    score2 = test(model, vali_loader)

    if max(score0,score1,score2) == score0:
        st.write("전시 입니다.")
    elif max(score0,score1,score2) == score1:
        st.write("야외 입니다.")
    elif max(score0,score1,score2) == score2:
        st.write("체험 입니다.")

    # st.write("결과")
    
    # st.write(image_file.name)
    # plt.imshow(vali_dataset.__getitem__(0))
    

    
