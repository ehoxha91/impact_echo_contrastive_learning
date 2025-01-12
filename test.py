import warnings
warnings.filterwarnings("ignore")
import sys

home_path ="/home/ehoxha/projects2023"
# home_path="/home/roboticslab/impact_echo_projects"
sys.path.insert(0, '../')
sys.path.insert(0, 'models')
sys.path.insert(0, 'weights')

import logging
import torch
# from models.echoFormer import EchoNet
# from models.classifier import Classifier
from utils import *
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_ds1, y_ds1 = load_ds1_test_data_into_torch_tensor(device=device)

normalize=False
softmax_use=True
args = sys.argv

if args[1] == 'supervised_contrastive_learning':
    
    from models.ienet_cl import EchoNet as EchoNet
    from models.ienet_cl import Classifier as Classifier
    model_name = args[2]
    path_for_results = 'supervised_contrastive_learning'
    logger.info(f"Using device: {device}")
    logger.info(f"Using supervised trained model: {model_name}")
    model = EchoNet(verbose=False).to(device)
    
    model.load_state_dict(torch.load(f'weights/{model_name}.pth'))
    logger.info('Trained model loaded')

    model.eval()
    logger.info('Model evaluation mode')

    # load classifier model
    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(f'weights/{model_name}_classifier.pth'))
    logger.info('Trained classifier loaded')
    classifier.eval()
    logger.info('Classifier on evaluation mode')

    # DS1 - Results 
    out1 = model(X_ds1, train=False)
    out1_for_tsne = out1.view(out1.size(1), out1.size(2))
    out1_for_tsne = out1_for_tsne.cpu().detach().numpy()
    if normalize:
        out1 = F.normalize(out1)
    out1 = classifier(out1)
    out1 = out1.view(out1.size(1), out1.size(2))
    if softmax_use:
        out1 = F.softmax(out1, dim=1)
    res1 = out1.cpu().detach().numpy()

    ax = plt.subplot()
    im = ax.imshow(np.reshape(res1[:,0], (9,28)), cmap='Spectral', interpolation='hamming')
    plt.axis("OFF")
    plt.savefig(f'defect_map_ds1.png', dpi=300)
    plt.show()
    print(f"DS1: ({np.min(res1[:,0])},{np.max(res1[:,0])})")
    logger.info("DS1 - Test Map Generated")
    
elif args[1] == 'self_supervised_contrastive_learning':
    
    from models.ienet import EchoNet as EchoNet
    from models.ienet import Classifier as Classifier
    model_name = args[2]
    path_for_results = 'self_supervised_contrastive_learning'
    logger.info(f"Using device: {device}")
    logger.info(f"Using supervised trained model: {model_name}")
    model = EchoNet(verbose=False).to(device)
    
    model.load_state_dict(torch.load(f'weights/{model_name}.pth'))
    logger.info('Trained model loaded')

    model.eval()
    logger.info('Model evaluation mode')

    # load classifier model
    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(f'weights/{model_name}_classifier.pth'))
    logger.info('Trained classifier loaded')
    classifier.eval()
    logger.info('Classifier on evaluation mode')

    # DS1 - Results 
    out1 = model(X_ds1, train=False)
    out1_for_tsne = out1.view(out1.size(1), out1.size(2))
    out1_for_tsne = out1_for_tsne.cpu().detach().numpy()
    if normalize:
        out1 = F.normalize(out1)
    out1 = classifier(out1)
    out1 = out1.view(out1.size(1), out1.size(2))
    if softmax_use:
        out1 = F.softmax(out1, dim=1)
    res1 = out1.cpu().detach().numpy()

    ax = plt.subplot()
    im = ax.imshow(np.reshape(res1[:,0], (9,28)), cmap='Spectral', interpolation='hamming')
    plt.axis("OFF")
    plt.savefig(f'defect_map_ds1_self_supervised_cl.png', dpi=300)
    plt.show()
    print(f"DS1: ({np.min(res1[:,0])},{np.max(res1[:,0])})")
    logger.info("DS1 - Test Map Generated")

else:
    logger.warning(f'Please use the correct input format one of the followings:')
    logger.warning(f"python3 test.py supervised_contrastive_learning supervised_contrastive_learning_model")
    logger.warning(f"python3 test.py self_supervised_contrastive_learning self_supervised_contrastive_learning_model")