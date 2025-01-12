import warnings
warnings.filterwarnings("ignore")

import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../dataloaders')
sys.path.insert(0, '../models')

from utils import *

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_ds1, y_ds1 = load_ds1_test_data_into_torch_tensor(
    device=device, 
    X_path='../data/X_test_860.npy', 
    y_path='../data/y_test.npy'
)
X_may, X_june = load_ccny_sep2022_data_into_torch_tensor(
    device=device,
    X_path='../data/X_our_slab_size860.npy'
)
# X_overlay, y_overlay = load_ds3_overlay_test_data_into_torch_tensor(device=device)

# No GT labels
X_nov23 = load_ccny_nov2023_data_into_torch_tensor2(
    device=device,
    X_path='../data/nov2023_non_resampled.npy'
)

model_name = 'baseline_model'
from models.ienet_baseline import IENet
path_for_results = 'baseline'
logger.info(f"Using device: {device}")
logger.info(f"Using unsupervised trained model: {model_name}")

classifier = IENet(verbose=False).to(device)
classifier.load_state_dict(torch.load(f'../weights/{model_name}.pth')) 
classifier.eval()

out1, _ = classifier(X_ds1)
out1 = out1.view(out1.size(1), out1.size(2))
out1 = F.softmax(out1, dim=1)
res1 = out1.cpu().detach().numpy()
ax = plt.subplot()
im = ax.imshow(np.reshape(res1[:,0], (9,28)), cmap='Spectral', interpolation='hamming')
plt.axis("OFF")
plt.savefig(f'{model_name}_ds1.png', dpi=300)
plt.show()
logger.info("DS1 - Test Map Generated")

out2, _ = classifier(X_may)
out3, _ = classifier(X_june)
out2 = out2.view(out2.size(1), out2.size(2))
out2 = F.softmax(out2, dim=1)
res2 = out2.cpu().detach().numpy()
out3 = out3.view(out3.size(1), out3.size(2))
out3 = F.softmax(out3, dim=1)
res3 = out3.cpu().detach().numpy()

ax = plt.subplot()
im = ax.imshow(np.reshape(res2[:,0], (31,38)), cmap='Spectral', interpolation='hamming')
plt.axis("OFF")
plt.savefig(f'{model_name}_ccny_p1.png', dpi=300)
plt.show()

ax = plt.subplot()
im = ax.imshow(np.reshape(res3[:,0], (19,34)), cmap='Spectral', interpolation='hamming')
plt.axis("OFF")
plt.savefig(f'{model_name}_ccny_p2.png', dpi=300)
plt.show()
logger.info("CCNY - Sep 2022 - Test Map Generated")

out4, _ = classifier(X_nov23)
out4 = out4.view(out4.size(1), out4.size(2))
out4 = F.softmax(out4, dim=1)
res4 = out4.cpu().detach().numpy()
ax = plt.subplot()
im = ax.imshow(np.reshape(res4[:,0], (44,34)), cmap='Spectral', interpolation='hamming')
plt.axis("OFF")
plt.savefig(f'{model_name}_ccny_nov2023.png', dpi=300)
plt.show()
logger.info("CCNY - Nov 2023 - Test Map Generated")