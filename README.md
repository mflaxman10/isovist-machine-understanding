# Machine Understanding of Architectural Space: From Analytical to Generative Applications   

This repository contains the code and models for the experiments conducted as part of the PhD research project "Machine Understanding of Architectural Space: From Analytical to Generative Applications" at EPFL Media and Design Lab (LDM). This project explores machine learning techniques for understanding architectural spaces from isovist representation, ranging from analytical tasks such as spatial typification and semantic discovery to generative tasks such as spatial sequence synthesis. The repository hosts multiple models developed during the research, along with the datasets and scripts to replicate the experiments.   
   
**Link to thesis > [EPFL Infoscience](https://infoscience.epfl.ch/handle/20.500.14299/241507)

## Author  
Mikhael Johanes

## Supervisor  
Jeffrey Huang


## Repository Structure

```bash
├── ae/                         # Vanilla autoencoder (ch2 : relational typology)
├── vae/                        # Variational autoencoder (ch2 : relational typology)
├── aae/                        # Adversarial autoencoder (misc: used as perceptual loss for gan inversion)
├── progan1d/                   # Progressive growing generative adversarial network (ch3: latent semantics)
├── latentgandiscovery/         # Unsupervised latent semantic discovery (ch3: latent semantics)
├── gist1/                      # Generative isovists transformer (ch4: latent discretization)
├── floorplan_isovist_dataset/  # Placeholder for the dataset (to be downloaded separately)
├── weights/                    # Placeholder for model weights (to be downloaded separately)
├── experiments/                # Folder for experiment results
├── notebooks/                  # Jupyter notebooks for demonstrations (coming soon)
├── requirements.txt            # Python dependencies
└── README.md                   # README
└── LICENSE                     # MIT license
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/johanesmikhael/isovist-machine-understanding.git
cd isovist-machine-understanding
```
2. Create and activate a Conda environment:
```bash
# Create a new environment with Python 3.x (replace x with the specific version if needed)
conda create --name arch-space-env python=3.8

# Activate the environment
conda activate arch-space-env
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset for the experiments can be downloaded from Zenodo. Follow these steps:

Go to [Zenodo](https://doi.org/10.5281/zenodo.13871782) and download the dataset.
Unzip the dataset and place the content in the floorplan_isovist_dataset/ folder.

## Models
The pre-trained model weights are available on HuggingFace. To use them:

Go to HuggingFace [repo](https://huggingface.co/mjohanes/isovist-ml) and download the model weights.
Place the weights in the weights/ folder.

## Training and sampling
We can train the models by using the given script [model_name]_train.py and passing the suitable configuration via --config argument
```bash
# train a vae model
python vae_train.py --config ./vae/conf/vae_1000k.json
```

By default all the results will be stored in experiment/ folder.

## Keywords
isovist, floor plan, machine learning, architecture