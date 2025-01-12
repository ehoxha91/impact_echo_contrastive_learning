import torch
import torch.nn as nn
from pytorch_metric_learning.losses import NTXentLoss

class HybridLossWithDualAug(nn.Module):

    def __init__(self, temperature=0.1, 
                 contrastive_weight = 0.5, 
                 classifier_1_weight = 0.25, 
                 classifier_2_weight = 0.25):
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.classifier_1_weight = classifier_1_weight
        self.classifier_2_weight = classifier_2_weight
        
        self.temperature = temperature
        self.contrastive_unsupervised = NTXentLoss(temperature=self.temperature)
        self.classifier_supervised = nn.CrossEntropyLoss()

    def forward(self, projections, pseudo_labels, predictions1, predictions2, labels):
        loss_contrastive = self.contrastive_unsupervised(projections, pseudo_labels)
        loss_classifier1 = self.classifier_supervised(predictions1, labels)
        loss_classifier2 = self.classifier_supervised(predictions2, labels)
        return self.contrastive_weight*loss_contrastive,  \
               self.classifier_1_weight*loss_classifier1, \
               self.classifier_2_weight*loss_classifier2
    
class HybridLossWithDualAugAndOriginal(nn.Module):

    def __init__(self, temperature=0.1, 
                 contrastive_weight = 0.3, 
                 classifier_1_weight = 1, 
                 classifier_2_weight = 1,
                 classifier_3_weight = 1):
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.classifier_1_weight = classifier_1_weight
        self.classifier_2_weight = classifier_2_weight
        self.classifier_3_weight = classifier_3_weight
        
        self.temperature = temperature
        self.contrastive_unsupervised = NTXentLoss(temperature=self.temperature)
        self.classifier_supervised = nn.CrossEntropyLoss()

    def forward(self, projections1, projections2, projections3, predictions1, predictions2, predictions3, labels):
        loss_contrastive1 = self.contrastive_unsupervised(projections1, labels)
        loss_contrastive2 = self.contrastive_unsupervised(projections2, labels)
        loss_contrastive3 = self.contrastive_unsupervised(projections3, labels)
        loss_classifier1 = self.classifier_supervised(predictions1, labels)
        loss_classifier2 = self.classifier_supervised(predictions2, labels)
        loss_classifier3 = self.classifier_supervised(predictions3, labels)
        return self.contrastive_weight*loss_contrastive1,  \
               self.contrastive_weight*loss_contrastive2,  \
               self.contrastive_weight*loss_contrastive3,  \
               self.classifier_1_weight*loss_classifier1,  \
               self.classifier_2_weight*loss_classifier2,  \
               self.classifier_3_weight*loss_classifier3
    
class HybridContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.1, 
                 contrastive_weight = 0.5, 
                 classifier_1_weight = 0.25, 
                 classifier_2_weight = 0.25):
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.classifier_1_weight = classifier_1_weight
        self.classifier_2_weight = classifier_2_weight
        
        self.temperature = temperature
        self.contrastive_unsupervised = NTXentLoss(temperature=self.temperature)
        self.classifier_supervised = nn.CrossEntropyLoss()

    def forward(self, projections, pseudo_labels, predictions1, predictions2, labels):
        loss_contrastive = self.contrastive_unsupervised(projections, pseudo_labels)
        loss_classifier1 = self.classifier_supervised(predictions1, labels)
        loss_classifier2 = self.classifier_supervised(predictions2, labels)
        loss =  self.contrastive_weight*loss_contrastive + \
               self.classifier_1_weight*loss_classifier1 + \
               self.classifier_2_weight*loss_classifier2
        return loss


class HybridLoss(nn.Module):

    def __init__(self, temperature=0.1, alpha=0.5):
        super().__init__()

        self.alpha = alpha
        self.temperature = temperature
        self.contrastive_unsupervised = NTXentLoss(temperature=self.temperature)
        self.classifier_supervised = nn.CrossEntropyLoss()

    def forward(self, projections, pseudo_labels, predictions, labels):
        loss_contrastive = self.contrastive_unsupervised(projections, pseudo_labels)
        loss_classifier = self.classifier_supervised(predictions, labels)
        return self.alpha*loss_contrastive, (1-self.alpha)*loss_classifier