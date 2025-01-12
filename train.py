import warnings
warnings.filterwarnings("ignore")
import sys
home_path="/home/ehoxha/projects2023"
sys.path.insert(0, f'{home_path}/')

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.dataloader import ImpactEchoDatasetCL, ImpactEchoDatasetClassifier, ImpactEchoDatasetBetter
from pytorch_metric_learning.losses import NTXentLoss, SignalToNoiseRatioContrastiveLoss
import tqdm
from utils import *
import torch.nn.functional as F

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
 
# set plot_defect_map to false to stop plotting the defect map while traning
plot_defect_map = True

args = sys.argv

if args[1] == 'supervised_contrastive_learning':

    from models.ienet_cl import EchoNet as EchoNet
    from models.ienet_cl import Classifier as Classifier

    name_of_the_class_model_used = EchoNet.__name__
    name_of_the_class_classifier_used = Classifier.__name__
    epochs = int(args[2])
    batch_size = 32
    model_name = 'supervised_contrastive_learning_model'
    learning_rate_cl = 0.0001
    learning_rate_classifier = 0.0001

    fh = logging.FileHandler(f'{model_name}.log')
    logger.addHandler(fh)
    logger.info(f"Training model: {model_name}")
    logger.info(f"Learning rates: contrastive {learning_rate_cl}, classifier {learning_rate_classifier}")
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Training batch size: {batch_size}")
    logger.info(f"Model used: {name_of_the_class_model_used}")
    logger.info(f"Classifier used: {name_of_the_class_classifier_used}")
  
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']

    dataset = ImpactEchoDatasetCL(X_path, y_path=y_path, sr = [500000, 500000], array_size=860, shuffle=True, augment_data=True)
    logger.info(f"Total number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=4)

    model = EchoNet(verbose=False).to(device=device)
    classifier = Classifier().to(device=device)

    contrastive_loss = NTXentLoss(temperature=0.07).to(device=device)
    contrastive_loss_snr = SignalToNoiseRatioContrastiveLoss().to(device=device)
    classification_loss = nn.CrossEntropyLoss().to(device=device)

    optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate_cl)
    optimizer2 = torch.optim.Adam(classifier.parameters(), lr=learning_rate_classifier)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)
    
    def train_hybrid_with_dual_augmented_signals():
        model.train()
        classifier.train()
        total_loss = 0.0
        for data in tqdm.tqdm(dataloader):
            aug_1 = data[0].to(device)
            aug_2 = data[1].to(device)
            x_original = data[3].to(device)
            
            aug_1 = aug_1.view(aug_1.size(0), 1, aug_1.size(1))
            aug_2 = aug_2.view(aug_2.size(0), 1, aug_2.size(1))
            x_original = x_original.view(x_original.size(0), 1, x_original.size(1))
            y_true = data[2].to(device, dtype=torch.int).long()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            embedding1 = model(aug_1, train=True)
            embedding2 = model(aug_2, train=True)
            embedding3 = model(x_original, train=True)

            y_preds3 = classifier(embedding3)
            y_preds3 = F.softmax(y_preds3, dim=1)
            y_preds3 = y_preds3.squeeze(0)

            projections = torch.cat((embedding1, embedding2), dim=1)
            # remove first dimension of projections
            projections = projections.view(projections.size(1), projections.size(2))
            labels = torch.cat((y_true, y_true))

            loss1 = contrastive_loss(projections, labels)
            loss2 = contrastive_loss_snr(projections, labels)
            loss3 = classification_loss(y_preds3, y_true) 
            beta = 0.3
            loss_combined = beta*(loss1 + loss2)+(1-beta)*loss3
            loss_combined.backward(retain_graph=True)

            # step the optimizer for model
            optimizer1.step()
            
            # update classifier weights
            optimizer2.zero_grad()
            loss3.backward()
            optimizer2.step()
            
            total_loss += loss1.item() + loss2.item() + loss3.item()
            
            writer.add_scalar('Supervised/NTXentLoss', loss1.item(), epoch)
            writer.add_scalar('Supervised/ContrastiveSNR', loss2.item(), epoch)
            writer.add_scalar('Supervised/Classification', loss3.item(), epoch)
            writer.add_scalar('Supervised/CombinedLoss', loss_combined.item(), epoch)
            writer.add_scalar('Supervised/TotalLoss', total_loss, epoch)

        return total_loss / len(dataset)


    dataset3 = ImpactEchoDatasetBetter()

    best_loss = float('inf')
    not_improved_count = 0
    for epoch in range(0, epochs):
        loss = train_hybrid_with_dual_augmented_signals()
        logger.info(f'Epoch {epoch:03d}, Loss: {loss:.8f}')
        model.eval()
        classifier.eval()
    
        if plot_defect_map:
            X_nov23 = load_ccny_nov2023_data_into_torch_tensor2(device=device)
            out4 = model(X_nov23, train=False)
            out4 = classifier(out4)
            out4 = out4.view(out4.size(1), out4.size(2))
            out4 = F.softmax(out4, dim=1)
            res4 = out4.cpu().detach().numpy()

            ax = plt.subplot()
            im = ax.imshow(np.reshape(res4[:,0], (44,34)), cmap='Spectral', interpolation='hamming')
            plt.axis("OFF")
            plt.savefig(f"results_{epoch}.png", dpi=300)
            plt.show()
            logger.info("CCNY - Nov 2023 - Test Map Generated")

        # validation loss computation
        val_loss = 0.0
        print("evaluation")
        with torch.no_grad():
            X_val, y_val = dataset3.__get_validation_data__()
            X_val = X_val.to(device)
            X_val = X_val.view(X_val.size(0), 1, X_val.size(1))
            y_val = y_val.to(device, dtype=torch.int).long()

            embedding1 = model(X_val, train=False)
            y_preds1 = classifier(embedding1)
            y_preds1 = y_preds1.squeeze(0)

            l = classification_loss(y_preds1, y_val)
            val_loss = l.item()
            
            if val_loss < best_loss:
                not_improved_count = 0
                logger.info(f"Best Loss: {val_loss:.8f}")
                best_loss = val_loss
                writer.add_scalar(f'Hybrid/{model_name}/Contrastive', val_loss, epoch)
                save_hybrid_model(embedding_model=model, 
                                classifier_model=classifier, 
                                path=f'weights/{model_name}')
            else:
                not_improved_count += 1    
                if not_improved_count > 20:
                    sys.exit(-1)
        scheduler1.step(loss)
        scheduler2.step(loss)

elif args[1] == 'self_supervised_contrastive_learning':
    
    from models.ienet import EchoNet as EchoNet
    from models.ienet import Classifier as Classifier
    
    name_of_the_class_model_used = EchoNet.__name__
    name_of_the_class_classifier_used = Classifier.__name__
    
    model_name = 'self_supervised_contrastive_learning_model'
    epochs = int(args[2])
    batch_size = 256

    fh = logging.FileHandler(f'{model_name}.log')
    logger.addHandler(fh)
    logger.info(f"Training model: {model_name}")
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Training batch size: {batch_size}")
    logger.info(f"Model used: {name_of_the_class_model_used}")
    logger.info(f"Classifier used: {name_of_the_class_classifier_used}")
  

    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']

    dataset = ImpactEchoDatasetCL(X_path, y_path=y_path, sr = [500000, 500000], array_size=860, shuffle=True, augment_data=True)
    logger.info(f"Total number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # This is similar to IENet model, but it has a projection head and 
    # no classifier while training using unsupervised learning (just contrastive loss: NTXentLoss)
    model = EchoNet(verbose=False).to(device)

    loss_function = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def train_contrastive_using_loss_unsupervised():
        model.train()
        total_loss = 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            batch_loss = 0
            optimizer.zero_grad()

            aug_1 = data[0].to(device)
            aug_2 = data[1].to(device)
            aug_1 = aug_1.view(aug_1.size(0), 1, aug_1.size(1))
            aug_2 = aug_2.view(aug_2.size(0), 1, aug_2.size(1))
            
            h1, z1 = model(aug_1, train=True)
            h2, z2 = model(aug_2, train=True)

            projections = torch.cat((z1, z2), dim=1)
            # remove first dimension of projections
            projections = projections.view(projections.size(1), projections.size(2))

            indices = torch.arange(0, z1.size(1), device=z2.device)
            labels = torch.cat((indices, indices))

            batch_loss = loss_function(projections, labels)
            batch_loss.backward()
            total_loss += batch_loss.item()
            
            optimizer.step()
            return total_loss / len(dataset)*1.0


    best_loss = float('inf')
    not_improved_count = 0
    stop_ucl = False
    for epoch in range(0, epochs):
        loss = train_contrastive_using_loss_unsupervised()
        logger.info(f'Epoch {epoch:03d}, Loss: {loss:.8f}')
        writer.add_scalar(f"CL Training Loss ({name_of_the_class_model_used, model_name})", loss, epoch)
        if loss < best_loss:
            not_improved_count = 0
            logger.info(f"Best Loss: {loss:.4f}")
            best_loss = loss
            save_model(model, f'weights/{model_name}.pth')
        else:
            not_improved_count += 1
            if not_improved_count > 15:
                stop_ucl = True
        if stop_ucl:
            break
        scheduler.step()

    writer.flush()
    writer.close()

    logger.info("Training Classifier")

    epochs = int(args[3])
    batch_size = 32

    dataset = ImpactEchoDatasetClassifier(X_path, y_path=y_path, array_size = 860)
    logger.info(f"Total number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Classifier layer
    classifier = Classifier().to(device)

    loss_classifier = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.eval()
    torch.backends.cudnn.enabled = False
    def train_classifier():
        total_loss = 0
        classifier.train()
        for data in tqdm.tqdm(dataloader):   
            optimizer.zero_grad()

            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))

            # CL trained model is used to get data representations
            y_representation = model(X, train=False)

            # get predictions from classifier using data representations
            output = classifier(y_representation)
            output = output.squeeze(0)

            loss = loss_classifier(output, labels)
            loss.backward()
            total_loss += loss.item()

            optimizer.step()

        return total_loss / len(dataset)

    logger.info('Training classifier...')
    best_loss = float('inf')
    not_improved_count = 0
    stop_ucl = False
    for epoch in range(0, epochs):
        loss = train_classifier()
        logger.info(f'Epoch {epoch:03d}, Loss: {loss:.8f}')
        writer.add_scalar(f"Supevised Classifier Loss ({name_of_the_class_model_used, model_name})", loss, epoch)
        if loss < best_loss:
            not_improved_count = 0
            logger.info(f"Best Loss: {loss:.4f}")
            best_loss = loss
            save_model(classifier, f'weights/{model_name}_classifier.pth')
        else:
            not_improved_count += 1
            if not_improved_count > 40:
                stop_ucl = True
        if stop_ucl:
            break
        scheduler.step()

    writer.flush()
    writer.close()

else:
    logger.warning(f'Please use the correct input format one of the followings:')
    logger.warning(f"python3 train.py supervised_contrastive_learning <epochs>")
    logger.warning(f"python3 train.py self_supervised_contrastive_learning <epochs_feature_extraction_model> <epochs_classifier>")