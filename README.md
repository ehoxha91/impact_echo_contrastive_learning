### Repository Description (LLM generated description)

This repository contains the code and datasets accompanying the paper **"Self-Supervised and Hybrid Learning Approaches for Robust Impact Echo Signal Interpretation"**. 

In this work, we propose two innovative learning-based methods for interpreting Impact Echo (IE) signals, a critical non-destructive evaluation (NDE) technique used to assess the structural integrity of concrete, such as in bridge decks. Traditional manual IE interpretation is labor-intensive, while automated methods often require expert calibration. Our proposed methods aim to address the challenges of generalization and robustness that have limited the effectiveness of learning-based IE approaches.

The key contributions of this work are:
1. **Self-Supervised Learning for IE**: A self-supervised learning approach using contrastive loss to leverage unlabeled data for feature extraction. This technique significantly reduces the need for labeled data and can be paired with simpler downstream methods like clustering.
   
2. **Hybrid Learning Strategy**: A hybrid approach combining contrastive and cross-entropy losses to improve the generalization and robustness of learning-based IE methods, ensuring better performance across varied experimental conditions.

3. **Datasets and Model Weights**: (to be added to this repository)

By sharing the code, datasets, and models, we aim to accelerate progress in autonomous IE signal interpretation, and enhance the generalization and reliability of NDE methods for concrete structures.

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/ehoxha91/impact_echo_contrastive_learning.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


### Usage:

#### Train Supervised: 
  ```bash
    python3 train.py supervised_contrastive_learning <epochs>
  ```
#### Train Hybrid:
  ```bash
      python3 train.py self_supervised_contrastive_learning <epochs_feature_extraction_model> <epochs_classifier>
  ```
#### To evaluate the models:
  ```bash
    python3 test.py supervised_contrastive_learning <model_name>
    python3 test.py self_supervised_contrastive_learning <model_name>
  ```
#### Datsets

Please refer to our other repository for dataset: https://github.com/ehoxha91/impact_echo_datasets

Datasets (DS1, DS2, DS3, and DS4 form the paper) are included in the repository.

#### Model Weights
The attached weights are generated only training for 1 epoch, and are there to serve only as a sample how to run our code, we will soon publish the original weights from our article.

#### Baseline Train/Test
Is Train/Test strategy from our previous work **"Robotic Inspection and Subsurface Defect Mapping Using Impact-Echo and Ground Penetrating Radar"** at RA-L. 
- Train: `python3 train_baseline.py <epochs>`
- Test: `python3 test_baseline.py`

### Citation

If you use something in this repository, please be kind and cite our work through:

```
@article{HOXHA2025139829,
title = {Contrastive learning for robust defect mapping in concrete slabs using impact echo},
journal = {Construction and Building Materials},
volume = {461},
pages = {139829},
year = {2025},
issn = {0950-0618},
doi = {https://doi.org/10.1016/j.conbuildmat.2024.139829},
url = {https://www.sciencedirect.com/science/article/pii/S0950061824049717},
author = {Ejup Hoxha and Jinglun Feng and Agnimitra Sengupta and David Kirakosian and Yang He and Bo Shang and Ardian Gjinofci and Jizhong Xiao},
keywords = {Impact echo, Bridge decks, Contrastive learning, Concrete defects}
}
```
and

```
@ARTICLE{10168232,
  author={Hoxha, Ejup and Feng, Jinglun and Sanakov, Diar and Xiao, Jizhong},
  journal={IEEE Robotics and Automation Letters}, 
  title={Robotic Inspection and Subsurface Defect Mapping Using Impact-Echo and Ground Penetrating Radar}, 
  year={2023},
  volume={8},
  number={8},
  pages={4943-4950},
  doi={10.1109/LRA.2023.3290386}}
```

### License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---