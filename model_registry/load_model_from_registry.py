# load_model_from_registry.py
import os
from argparse import ArgumentParser

import mlflow
import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image

# 0. set mlflow environments
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 1. load model from mlflow
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
parser.add_argument("--run-id", dest="run_id", type=str)
args = parser.parse_args()

model_pipeline = mlflow.pytorch.load_model(f"runs:/{args.run_id}/{args.model_name}")

# 2. get data
df = pd.read_csv("data.csv")

X = np.array(df["image_path"])
y = np.array(df["target"])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2022)

file_path = '../api_serving/monday/'

BATCH_SIZE = 64
EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(file_path + img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = CustomDataset(file_paths=X_train, labels=y_train, transform=data_transform)
test_dataset = CustomDataset(file_paths=X_test, labels=y_test, transform=data_transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 2. model development and train
# model = models.mobilenet_v2(pretrained=True)

# for parameter in model.parameters():
#     parameter.requires_grad = False

# num_features = model.classifier[1].in_features

# model.classifier[1] = nn.Linear(num_features, 5)

model_pipeline = model_pipeline.to(DEVICE)

optimizer = torch.optim.Adam(model_pipeline.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : 0.95 ** epoch)

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE, dtype=torch.float)
            label = label.to(DEVICE)
            output = model(image).squeeze(dim=1)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, EPOCHS + 1):
    valid_loss, valid_accuracy = evaluate(model_pipeline, test_dataloader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, valid_loss, valid_accuracy))

# train_pred = []
# for i in range(len(train_pred_tensor)):
#   train_pred.append(np.transpose(np.array(train_pred_tensor[i]))[0])


# print("Train Accuracy :", train_accuracy)
print("Test Accuracy :", valid_accuracy)