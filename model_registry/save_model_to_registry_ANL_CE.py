# save_model_to_registry_ANL_CE.py

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

parser = ArgumentParser()
parser.add_argument("--dataset", "-d", dest="dataset", type=str, required=True, help="specify the dataset")
parser.add_argument("--label", "-l", dest="label", type=str, choices=["ground_truth", "user_feedback"], default="ground_truth", help="Choose whether to use the ground truth label or the user feedback label")
parser.add_argument("--user_accuracy", dest="user_accuracy", type=float, default=0.7, help="select the accuracy of user feedback")
parser.add_argument("--monday_num", dest="monday_num", type=int, required=True, help="specify the number for monday")
parser.add_argument("--tuesday_num", dest="tuesday_num", type=int, required=False, help="specify the number for tuesday")
parser.add_argument("--model-name", dest="model_name", type=str, default="cls_model", help="specify the model name (default: cls_model)")
args = parser.parse_args()

print("===================================================================")
print("Comparative Experiments: ANL_CE")
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

class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ActiveNegativeLoss(torch.nn.Module):
    def __init__(self, active_loss, negative_loss,
                 alpha=1., beta=1., delta=0.) -> None:
        super().__init__()
        self.active_loss = active_loss
        self.negative_loss = negative_loss
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
    
    def forward(self, pred, labels, model):
        al = self.active_loss(pred, labels)
        nl = self.negative_loss(pred, labels)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        
        loss = self.alpha * al + self.beta * nl + self.delta * l1_norm
        
        return loss

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()
    
class NormalizedNegativeCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, min_prob=1e-7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.min_prob = min_prob
        self.A = - torch.tensor(min_prob).log()
    
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1)
        pred = self.A + pred.log() # - log(1e-7) - (- log(p(k|x)))
        label_one_hot = F.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()
    
class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, num_classes=10):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss / normalizor

        return loss.mean()

class NormalizedNegativeFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma=1e-3, min_prob=1e-7) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.min_prob = min_prob
        self.logmp = torch.tensor(self.min_prob).log()
        self.A = - (1 - min_prob)**gamma * self.logmp
    
    def forward(self, input, target):
        logmp = self.logmp.to(input.device)
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1).clamp(min=logmp)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = 1 - (self.A - loss) / (self.num_classes * self.A - normalizor)
        return loss.mean()

def _anl(active_loss, negative_loss):
    return ActiveNegativeLoss(active_loss,
                              negative_loss
                              )

def nce(num_classes):
    return NormalizedCrossEntropy(num_classes)

def nnce(num_classes):
    return NormalizedNegativeCrossEntropy(num_classes)

def nfl(num_classes):
    return NormalizedFocalLoss(num_classes)

def nnfl(num_classes):
    return NormalizedNegativeFocalLoss(num_classes)

def anl_ce(num_classes):
    return _anl(nce(num_classes), nnce(num_classes))

def anl_fl(num_classes):
    return _anl(nfl(num_classes), nnfl(num_classes))

criterion = anl_ce(class_num)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : 0.95 ** epoch)

def train(model, train_loader, optimizer, log_interval):
    model.train()
    correct = 0
    pred = []

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE, dtype=torch.float)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(image).squeeze(dim=1)
        loss = criterion(output, label, model)
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
    return train_accuracy, pred

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE, dtype=torch.float)
            label = label.to(DEVICE)
            output = model(image).squeeze(dim=1)
            test_loss += criterion(output, label, model).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, EPOCHS + 1):
    train_accuracy, train_pred_tensor = train(model, train_dataloader, optimizer, log_interval = 5)
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