
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../dataloaders')
sys.path.insert(0, '../models')
sys.path.insert(0, '../data')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.dataloader import ImpactEchoDatasetClassifier
import tqdm
from utils import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# configure logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from models.ienet_baseline import IENet

X_path = ['../data/X_train_860.npy']
y_path = ['../data/y_train.npy']

epochs = int(sys.argv[1])
model_name = 'baseline_model'
batch_size = 32
dataset = ImpactEchoDatasetClassifier(X_path, y_path=y_path, array_size=860)
print(f"Total number of training samples: {len(dataset)}")
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Classifier layer
classifier = IENet(verbose=False).to(device)

loss_classifier = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def train_classifier():
    total_loss = 0
    
    for data in tqdm.tqdm(dataloader):
        classifier.train()
        optimizer.zero_grad()

        X = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.int).long()
        X = X.view(X.size(0), 1, X.size(1))

        output = classifier(X)
        output = output[0].squeeze(0)

        loss = loss_classifier(output, labels)
        loss.backward()
        total_loss += loss.item()

        optimizer.step()

    return total_loss / len(dataset)

print('Training classifier...')

best_loss = float('inf')
for epoch in range(0, epochs):
    loss = train_classifier()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    writer.add_scalar("Loss/train", loss, epoch)
    if loss < best_loss:
        print(f"Best Loss: {loss:.4f}")
        best_loss = loss
        save_model(classifier, f'../weights/{model_name}.pth')
    scheduler.step()

writer.flush()
writer.close()