# save_model_to_registry.py

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
parser.add_argument("--target_day", "-t", dest="target_day", type=str, choices=["monday", "tuesday"], default="monday", help="specify the target day (default: monday)")
parser.add_argument("--label", "-l", dest="label", type=str, choices=["ground_truth", "user_feedback"], default="ground_truth", help="Choose whether to use the ground truth label or the user feedback label")
parser.add_argument("--user_accuracy", dest="user_accuracy", type=float, default=0.7, help="select the accuracy of user feedback")
parser.add_argument("--bayesian_cut_off", "-b", dest="bayesian_cut_off", type=str, choices=["Y", "N"], default="N", help="Choose whether to use the Bayesian cut-off (Y/N)")
parser.add_argument("--sampling_type", "-s", dest="sampling_type", type=str, choices=["none", "random", "l1_norm", "l2_norm", "cosine_similarity"], default="none", help="specify the sampling type (default: none)")
parser.add_argument("--monday_num", dest="monday_num", type=int, required=True, help="specify the number for monday")
parser.add_argument("--tuesday_num", dest="tuesday_num", type=int, required=False, help="specify the number for tuesday")
parser.add_argument("--ratio", "-r", dest="ratio", type=float, help="select the ratio to extract from the buffer")
parser.add_argument("--model-name", dest="model_name", type=str, default="cls_model", help="specify the model name (default: cls_model)")
args = parser.parse_args()

if args.target_day == "tuesday" and args.tuesday_num is None:
    parser.error("--tuesday_num must be provided when --target_day is 'tuesday'.")
if args.target_day == "tuesday" and args.sampling_type != "none" and args.ratio is None:
    parser.error("--ratio must be provided when --target_day is 'tuesday' and --sampling_type is not 'none'.")

print("===================================================================")
print(f"dataset         : {args.dataset}")
print(f"target_day      : {args.target_day}")
if args.target_day == "tuesday":
    print(f"label           : {args.label}", end= "")
    if args.label == "user_feedback":
        print(f" (accuracy {args.user_accuracy})")
    else:
        print("")
print(f"bayesian cut-off: {args.bayesian_cut_off}")
if args.target_day == "tuesday":
    print(f"sampling_type   : {args.sampling_type}")
    if args.sampling_type != "none":
        print(f"ratio           : {args.ratio}")
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

if args.target_day == "monday":
    # Monday Dataset
    X_train = X_train1
    y_train = y_train1
    X_test = X_test1
    y_test = y_test1

elif args.target_day == "tuesday":
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

    if args.sampling_type == "none":
        X_train = X_train2
        y_train = y_train2
        X_test = X_test1 + X_test2
        y_test = y_test1 + y_test2

    elif args.sampling_type == "random":
        random_indices = random.sample(range(len(X_train1)), int(len(X_train1)*args.ratio))

        X_train1 = [X_train1[i] for i in random_indices]
        y_train1 = [y_train1[i] for i in random_indices]

        X_train = X_train1 + X_train2
        y_train = y_train1 + y_train2
        X_test = X_test1 + X_test2
        y_test = y_test1 + y_test2
    
    elif args.sampling_type == "l1_norm" or "l2_norm" or "cosine_similarity":
        def l1_norm(a, b):
            add_inv_b = torch.mul(b, -1)
            summation = torch.add(a, add_inv_b)
            abs_val = torch.abs(summation)
            return torch.sum(abs_val).item()

        def l2_norm(a, b):
            add_inv_b = torch.mul(b, -1)
            summation = torch.add(a, add_inv_b)
            square = torch.mul(summation, summation)
            sqrt = torch.sqrt(square)
            return torch.sum(sqrt).item()

        def cosine_similarity(a, b):
            cos = torch.nn.functional.cosine_similarity(a, b)
            return torch.mean(cos).item()

        if args.sampling_type == "l1_norm":
            function = l1_norm
        elif args.sampling_type == "l2_norm":
            function = l2_norm
        else:
            function = cosine_similarity
        
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def load_and_preprocess_image(image_path):
            text_str = str(image_path.split('/')[3])
            image_path = os.path.join(image_path)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=text_str, images=image, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)

            return outputs.image_embeds.detach().cpu().numpy()
        
        X = np.array(X_train1)
        y = np.array(y_train1)

        img = []
        for k in range(class_num):
            img.append([])

        for i in range(len(X)):
            img[y[i]].append(X[i])

        selected_pair_file_list = []
        selected_pair_index_list = []

        for i in range(int(len(X)*args.ratio/2)):
            num_list = [num for num in range(class_num)]
            
            random_list = [None for num in range(class_num)]

            for k in range(class_num):
                if len(img[k]) > 0:
                    random_list[k] = random.sample(img[k], 1)[0]
                    img[k].remove(random_list[k])
                else:
                    num_list.remove(k)
            
            anchor_num = random.sample(num_list, 1)[0]
            num_list.remove(anchor_num)
            
            if len(num_list) == 0:
                parser.error("--ratio is too high.")

            dist = []
            dist_num = []

            for j in num_list:
                dist_value = function(torch.tensor(load_and_preprocess_image(random_list[j])), torch.tensor(load_and_preprocess_image(random_list[anchor_num])))
                dist.append(dist_value)
                dist_num.append(j)
            
            dist = np.array(dist)
            if args.sampling_type == "cosine_similarity":
                closest_index = dist_num[np.argmax(dist)]
            else:
                closest_index = dist_num[np.argmin(dist)]

            closest_path = random_list[closest_index]
            anchor_path = random_list[anchor_num]

            selected_pair_file_list.append(closest_path)
            selected_pair_file_list.append(anchor_path)

            selected_pair_index_list.append(closest_index)
            selected_pair_index_list.append(anchor_num)

        X_train1 = list(selected_pair_file_list)
        y_train1 = list(selected_pair_index_list)

        X_train = X_train1 + X_train2
        y_train = y_train1 + y_train2
        X_test = X_test1 + X_test2
        y_test = y_test1 + y_test2

print(f'X_train: {len(X_train)}, X_test: {len(X_test)}')

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
        if args.target_day == 'tuesday' and args.label == 'user_feedback' and generate_value() == 0:
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

# Bayesian Cut-off
if args.bayesian_cut_off == "Y":
    def calculate_uncertainty(predictions):
        # Calculate uncertainty as the variance of predictions
        return np.var(predictions, axis=0)

    def get_predictions_with_uncertainty(model, dataloader, num_samples=10):
        model.eval()
        all_outputs = []

        with torch.no_grad():
            for i in range(num_samples):
                outputs = []
                for images, _ in dataloader:
                    images = images.to(DEVICE, dtype=torch.float)
                    output = model(images).cpu().numpy()
                    output = np.sum(output, axis=1)
                    outputs.append(output)
                all_outputs.append(np.concatenate(outputs))
        print("")
        all_outputs = np.array(all_outputs)
        mean_outputs = np.mean(all_outputs, axis=0)
        uncertainties = calculate_uncertainty(all_outputs)
        
        return mean_outputs, uncertainties
    
    class BayesianDropout(nn.Dropout):
        def forward(self, x):
            return F.dropout(x, self.p, training=True, inplace=self.inplace)

    class BayesianMobileNetV2(nn.Module):
        def __init__(self, dropout_prob=0.5):
            super(BayesianMobileNetV2, self).__init__()
            mobilenet_v2 = models.mobilenet_v2(pretrained=True)
            
            # Extracting features part of MobileNetV2
            self.features = mobilenet_v2.features

            # Adding Bayesian Dropout layers in between the features
            self.bayesian_features = nn.Sequential()
            for idx, layer in enumerate(self.features):
                self.bayesian_features.add_module(f"layer_{idx}", layer)
                if isinstance(layer, nn.Conv2d):
                    self.bayesian_features.add_module(f"dropout_{idx}", BayesianDropout(p=dropout_prob))

            # Classifier part
            num_features = mobilenet_v2.classifier[1].in_features
            self.classifier = nn.Sequential(
                BayesianDropout(p=dropout_prob),
                nn.Linear(num_features, class_num)
            )

        def forward(self, x):
            x = self.bayesian_features(x)
            x = x.mean([2, 3])  # Global average pooling
            x = self.classifier(x)
            return x


    # Instantiate the Bayesian model
    model = BayesianMobileNetV2(dropout_prob=0.5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Start Bayesian Cut-off
    num_epochs = 10
    model.train()
    print("[Bayesian Cut-off]")
    for epoch in range(num_epochs):
        for images, labels in train_dataloader:
            images = images.to(DEVICE, dtype=torch.float)
            labels = labels.to(DEVICE, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')  
    print("")

    # Evaluate the model and remove outliers based on uncertainty
    mean_outputs, uncertainties = get_predictions_with_uncertainty(model, train_dataloader)

    # Calculate and exclude outliers based on a threshold
    threshold = np.percentile(uncertainties, 90)  # Example threshold (90th percentile)
    clean_train_indices = np.where(uncertainties <= threshold)[0]

    train_dataloader = DataLoader(
        CustomDataset_test([X_train[i] for i in clean_train_indices], [y_train[i] for i in clean_train_indices], transform=data_transform),
        batch_size=BATCH_SIZE, shuffle=True
    )

    model_pipeline = BayesianMobileNetV2(dropout_prob=0.5).to(DEVICE)
else:
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

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE, dtype=torch.float)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(image).squeeze(dim=1)
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
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

print("[Training]")
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