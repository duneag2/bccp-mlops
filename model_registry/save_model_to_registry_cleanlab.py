# save_model_to_registry_cleanlab.py

import os
from argparse import ArgumentParser

import mlflow
import numpy as np
import pandas as pd
import random
import psycopg2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from cleanlab.pruning import get_noise_indices
from cleanlab.classification import LearningWithNoisyLabels

parser = ArgumentParser()
parser.add_argument("--dataset", "-d", dest="dataset", type=str, required=True, help="specify the dataset")
parser.add_argument("--label", "-l", dest="label", type=str, choices=["ground_truth", "user_feedback"], default="ground_truth", help="Choose whether to use the ground truth label or the user feedback label")
parser.add_argument("--user_accuracy", dest="user_accuracy", type=float, default=0.7, help="select the accuracy of user feedback")
parser.add_argument("--monday_num", dest="monday_num", type=int, required=True, help="specify the number for monday")
parser.add_argument("--tuesday_num", dest="tuesday_num", type=int, required=False, help="specify the number for tuesday")
parser.add_argument("--model-name", dest="model_name", type=str, default="cls_model", help="specify the model name (default: cls_model)")
args = parser.parse_args()

print("===================================================================")
print("Comparative Experiments: Cleanlab")
print(f"dataset      : {args.dataset}")
print(f"target_day   : tuesday")
print(f"label        : {args.label}", end= "")
if args.label == "user_feedback":
    print(f" (accuracy {args.user_accuracy})")
else:
    print("")
print("===================================================================")

# 0. set mlflow environments
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 1. get data
db_connect = psycopg2.connect(
    user="myuser",
    password="mypassword",
    host="localhost",
    port=5432,
    database="mydatabase",
)

# Monday Dataset Original
df1 = pd.read_sql(f"SELECT * FROM {args.dataset} WHERE id <= {args.monday_num}", db_connect)

X = np.array(df1["image_path"])
y = np.array(df1["target"])

for i in range(len(X)):
    X[i] = f'../api_serving/{args.dataset}/monday/' + X[i]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, train_size=0.8, random_state=2022)

X_train1 = list(X_train1)
y_train1 = list(y_train1)
X_test1 = list(X_test1)
y_test1 = list(y_test1)

class_num = len(np.unique(y))

# Tuesday Dataset Original
df2 = pd.read_sql(f"SELECT * FROM {args.dataset} WHERE id BETWEEN {args.monday_num + 1} and {args.monday_num + args.tuesday_num}", db_connect)

X2 = np.array(df2["image_path"])
y2 = np.array(df2["target"])

for i in range(len(X2)):
    X2[i] = f'../api_serving/{args.dataset}/tuesday/' + X2[i]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, train_size=0.8, random_state=2022)

X_train2 = list(X_train2)
y_train2 = list(y_train2)
X_test2 = list(X_test2)
y_test2 = list(y_test2)

X_train = X_train2
y_train = y_train2
X_test = X_test1 + X_test2
y_test = y_test1 + y_test2

BATCH_SIZE = 64
EPOCHS = 15
USER_ACCURACY = args.user_accuracy
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_value():
    # Returns 1 with a probability of USER_ACCURACY, otherwise returns 0
    return random.choices([1, 0], weights=[USER_ACCURACY, 1-USER_ACCURACY], k=1)[0]

def select_random_except_label(label):
    # Creates a list according to the number of classes and removes the number corresponding to the label
    numbers = [num for num in range(class_num)]
    numbers.remove(label)
    
    # Select a random element from the list
    return random.choice(numbers)

class CustomDataset_train(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # label
        label = self.labels[idx]
        if args.label == 'user_feedback' and generate_value() == 0:
            label = select_random_except_label(label)

        return image, label

class CustomDataset_test(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = CustomDataset_train(file_paths=X_train, labels=y_train, transform=data_transform)
test_dataset = CustomDataset_test(file_paths=X_test, labels=y_test, transform=data_transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 2. model development and train
model = models.mobilenet_v2(pretrained=True)

for parameter in model.parameters():
    parameter.requires_grad = False

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(num_features, class_num)

model_pipeline = model.to(DEVICE)

optimizer = torch.optim.Adam(model_pipeline.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : 0.95 ** epoch)

def train(model, train_loader, optimizer, log_interval):
    model.train()
    correct = 0
    pred = []
    output_prob = []

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE, dtype=torch.float)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(image).squeeze(dim=1)
        output_prob.append(output)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        prediction = output.max(1, keepdim = True)[1]
        correct += prediction.eq(label.view_as(prediction)).sum().item()
        pred.append(prediction)

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
    train_accuracy = 100. * correct / len(train_loader.dataset)
    scheduler.step()
    return train_accuracy, pred, output_prob

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
    train_accuracy, train_pred_tensor, train_output_prob = train(model, train_dataloader, optimizer, log_interval = 5)

array_list = [tensor.cpu().detach().numpy() for tensor in train_output_prob]
concatenated_array = np.concatenate(array_list, axis=0)


noise_idx = get_noise_indices(s=y_train, psx = concatenated_array, confident_joint=None)

# # Use LearningWithNoisyLabels to train with noisy labels
# cleanlab_trainer = LearningWithNoisyLabels(clf=model_pipeline, seed=42)
# cleanlab_trainer.fit(X_train=X_train, y_train=y_train, noise_indices=noise_idx)

class CustomDataset_train_denoising(Dataset):
    def __init__(self, file_paths, labels, noise_indices, transform=None):
        self.file_paths = [file_paths[idx] for idx, is_noise in enumerate(noise_indices) if is_noise]
        self.labels = [labels[idx] for idx, is_noise in enumerate(noise_indices) if is_noise]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset instance with noise filtering
custom_dataset = CustomDataset_train_denoising(file_paths=X_train, labels=y_train, noise_indices=noise_idx, transform=data_transform)
train_dataloader_denoising = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


for epoch in range(1, EPOCHS + 1):
    train_accuracy, train_pred_tensor, train_output_prob = train(model, train_dataloader_denoising, optimizer, log_interval = 5)
    valid_loss, valid_accuracy = evaluate(model, test_dataloader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, valid_loss, valid_accuracy))

valid_loss, valid_accuracy = evaluate(model, test_dataloader)
print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
    epoch, valid_loss, valid_accuracy))

train_pred = []
for i in range(len(train_pred_tensor)):
  train_pred.append(np.transpose(np.array(train_pred_tensor[i].cpu()))[0])

train_pred = np.concatenate(train_pred)

print("Train Accuracy :", train_accuracy)
print("Test Accuracy :", valid_accuracy)

# 3. save model
mlflow.set_experiment("new-exp")

signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
input_sample = X_train[:10]

with mlflow.start_run():
    mlflow.log_metrics({"train_acc": train_accuracy, "test_acc": valid_accuracy})
    mlflow.pytorch.log_model(
        pytorch_model=model, 
        artifact_path=args.model_name, 
        signature=signature, 
        input_example=input_sample)

# 4. save data
df1.to_csv("data.csv", index=False)