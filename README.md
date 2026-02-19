# 🌸 Flower Recognition Project

A deep learning project for flower species recognition using PyTorch and Streamlit.
[Presentation](https://www.canva.com/design/DAHAc-KN7gQ/revzIBL_wn3lo8PMUp3jdw/view?utm_content=DAHAc-KN7gQ&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h8c2d3eea85)

## 📋 Project Overview

This project implements a ResNet9 (Residual Network) architecture to classify different flower species. The model is trained using PyTorch and deployed as an interactive web application using Streamlit.

### 🌼 Recognized Flower Species

The model can recognize the following 5 flower types:
- 🌼 **Daisy**
- 🌻 **Dandelion**
- 🌹 **Rose**
- 🌻 **Sunflower**
- 🌷 **Tulip**

## 🌐 Live Demo

Access the deployed application here: [https://flower-recognization.streamlit.app/](https://flower-recognization.streamlit.app/)

## 📁 Project Structure

```
apple-pie-h/Flower-Recognization-using-DeepLearning
├── images/                          # Sample test images for the model
├── flower_recognization.ipynb       # Jupyter notebook with model training code
├── flower.pth                       # Trained model weights (exported from notebook)
├── web.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Kaggle account (for dataset access)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/apple-pie-h.git
   cd apple-pie-h
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Setting Up Kaggle API

To download the dataset in the Jupyter notebook, you'll need to configure your Kaggle API credentials:

1. **Create a Kaggle account** at [kaggle.com](https://www.kaggle.com) if you don't have one

2. **Get your API key:**
   - Go to your Kaggle account settings: https://www.kaggle.com/settings
   - Scroll down to the "API" section
   - Click "Generate New Token"
   - This will provide you API token (save it, dont share it with anyone else)

3. **Configure the API key:**
   - While running the notebook cell, it will ask for Kaggle Uername, you may enter any random letter combination for it and press enter.
   - Then it will ask for API token, paste the API token you just created.


## 🔧 Usage

### Training the Model

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook flower_recognization.ipynb
   ```

2. Run all cells to:
   - Download the dataset using `opendatasets`
   - Preprocess the images
   - Train the CNN model
   - Export the trained model as `flower.pth`

### Running the Web Application

#### Local Deployment

```bash
streamlit run web.py
```

The app will open in your browser at `http://localhost:8501`

#### Using the Application

1. Upload a flower image using the file uploader
2. You can use:
   - Sample images from the `images/` folder
   - Your own flower images (`.jpg`)
   - Google-downloaded flower images
3. **Important:** Ensure the image contains one of the supported flowers (Daisy, Dandelion, Rose, Sunflower, or Tulip)
4. The model will predict the flower species
5. Results will be displayed with confidence scores

## 📦 Dependencies

```
streamlit          # Web application framework
torch              # Deep learning framework
torchvision        # Computer vision utilities
Pillow             # Image processing
numpy              # Numerical computations
matplotlib         # Visualization
opendatasets       # Dataset downloading
kaggle             # Kaggle API client
```

## 🧠 Model Details

- **Architecture:** ResNet9 (Residual Network with 9 layers)
- **Framework:** PyTorch
- **Model File:** `flower.pth`
- **Input:** RGB images of flowers
- **Output:** Flower species classification (5 classes: Daisy, Dandelion, Rose, Sunflower, Tulip)
- **Classes:** 5 flower species

## 🖼️ Testing Images

The `images/` folder contains sample flower images for testing the model. You can also use your own images downloaded from Google or other sources in `.jpg` format.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created using Colab by **apple-pie-h**

## 🙏 Acknowledgments

- Dataset sourced from Kaggle
- Built with PyTorch and Streamlit
- Deployed on Streamlit Cloud

---

**Note:** Make sure to have your Kaggle API credentials configured before running the training notebook to download the dataset successfully.
