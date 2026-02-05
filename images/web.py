import streamlit as st
import torch
import torchvision.transforms as tt
from PIL import Image
import torch.nn as nn

st.title("Flower Recognization Using Deep Learning (ResNet9 model)")
st.header("The model can recognize: Daisy, Dandelion, Rose, Sunflower, Tulip")


# Model definition
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ImageClassificationBase(nn.Module):
    pass


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)


# loading the model
@st.cache_resource
def load_model():
    model = ResNet9(3, 5)
    model.load_state_dict(torch.load("flower.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


model = load_model()

classes = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]


# image preprocessing definition
def preprocess_image(image):
    transform = tt.Compose([tt.Resize((64, 64)), tt.ToTensor()])
    return transform(image).unsqueeze(0)


st.write(
    "Upload an image of a flower to recognize its type (please make sure the file is in .jpg format)."
)

uploaded_file = st.file_uploader("Choose a file", type=["jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
        st.success(f"Prediction: {classes[pred]}")
