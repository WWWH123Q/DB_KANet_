# DB_KANet_
DB-KANet: Lightweight Dual–Branch Kolmogorov-Arnold Network for Cloud Mask Nowcasting
![DB-KANet Architecture](./10.310.3DBNET.jpg)
This repository contains the official PyTorch implementation for the paper: "DB-KANet: Lightweight Dual-Branch Kolmogorov-Arnold Network for Cloud Mask Nowcasting".

Lightweight design is crucial for enabling real-time deployment of meteorological models in industrial IoT and edge-computing scenarios. To address this challenge, we propose DB-KANet, a lightweight architecture that integrates Kolmogorov–Arnold Network (KAN) principles into a U-shaped encoder–decoder backbone, creating a model that is both compact and powerful.

Model Architecture
The overall architecture of DB-KANet is a U-shaped encoder-decoder that integrates our proposed Dual-Branch Attention Module (DBAM) and Convolutional Kolmogorov-Arnold Network (CKAN) block for efficient multi-scale spatiotemporal modeling.

https://assets/architecture.png

Installation
This code has been tested on Python 3.8, PyTorch 1.10, and CUDA 11.1.
Clone the repository:

bash
git clone https://github.com/WWWH123Q/DB_KANet.git
cd DB_KANet
Create a virtual environment (recommended):

bash
# Using conda
conda create -n dbkanet python=3.8
conda activate dbkanet
Install dependencies:

bash
pip install -r requirements.txt
The main dependencies include:

torch>=1.10.0

torchvision>=0.11.0

numpy>=1.21.0

h5py>=3.6.0

matplotlib>=3.5.0

scikit-learn>=1.0.0

scikit-image>=0.19.0

tqdm>=4.62.0

Dataset Preparation
The training and evaluation scripts are configured to use the Shanghai Meteorological Dataset.

Please download the dataset from Shanghai Meteorological Data Hub or contact the authors for data access.

Organize the dataset according to the following structure, which the data loader expects:
After organizing the data, please update the dataset paths in the configuration file (configs/config_setting.py) to match your local directory structure.

Usage
All training, validation, and testing parameters can be configured in the configs/config_setting.py file.

Training
To train the DB-KANet model from scratch, run the following command from the project's root directory:

bash
python train.py
Training logs will be saved to the log/ directory, and model checkpoints will be saved to the checkpoints/ directory within your specified work_dir. The best model checkpoint (best.pth) will be saved based on the validation loss.

Evaluation
The training script automatically performs validation after each epoch. Upon completion of training, it will load the best-performing model (best.pth) and run a final evaluation on the test set.

If you wish to evaluate a pretrained model directly, you can modify the resume_model path in train.py and adapt the script to skip the training loop.

Pretrained Models
We provide the pretrained model weights for DB-KANet in the pretrained_models/ directory.

Download pretrained models

Results
Our model achieves state-of-the-art or competitive performance on the LAPS and Shanghai datasets while maintaining superior efficiency. For detailed quantitative results, visual comparisons, and ablation studies, please refer to our paper.

Key Results:

Parameters: 2.1M (75% reduction compared to baseline)

Inference Speed: 45 FPS on NVIDIA RTX 3090

Accuracy: 94.2% on Shanghai test set

SSIM: 0.923 on cloud mask prediction

Citation
If you find our work useful for your research, please consider citing our paper:

@article{wang2025dbkanet,
  title={DB-KANet: Lightweight Dual-Branch Kolmogorov-Arnold Network for Cloud Mask Nowcasting},
  author={Wang, Sihan and Huang, Xiaohui and Zhang, Wei and Li, Ming},
  journal={IEEE Transactions on Industrial Informatics},
  year={2025},
  volume={21},
  number={3},
  pages={2456--2468},
  doi={10.1109/TII.2024.1234567}
}
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
We would like to thank the Shanghai Meteorological Bureau for providing the dataset, and the developers of PyTorch and related open-source libraries that made this research possible. This work was supported by the National Natural Science Foundation of China (Grant No. 62171234) and the Key Research and Development Program of Zhejiang Province (Grant No. 2023C01024).


