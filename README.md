# KEAG: Exploiting Kernel-based Embeddings, Attention and Geometric Properties for Improved Disk-based R-tree Insertions

We propose KEAG, a novel framework that exploits random Fourier features for representative points, attention mechanism, and geeometric properties to intelligently select appropriate candidates for R-tree insertions. KEAG models spatial relationships between candidate configurations and their surrounding context through an attention architecture enhanced by a kernel-based point embedder. It can reasonably integrate more informative features, significantly improving the quality of decisions for insertions.

## Requirements

```bash
- Python 3.12+
- Torch 2.1+, numpy, scipy
- gcc 13.3+, cmake 3.28+
```

## Dataset
We have provided a small Gaussian dataset for testing, located at 'KEAG/data/guassian/2M'.

## Environment setup
We use the open-source libspatialindex library as the R-tree backend, with custom modifications to support specialized ChooseSubtree and Split operations. 

For reproducibility, we recommend running the code and scripts inside a Docker container with Conda. For instance, in an Ubuntu-based Docker container, you can set up the environment as follows:
```bash
- cd KEAG
- apt install cmake
- conda create -n py312 -c conda-forge -y
- conda activate py312
- conda install python=3.12
- pip install -r requirements.txt
```
Next, compile libspatialindex by running the following commands:
```bash
- cd libspatialindex
- mkdir build; cd build
- cmake ..
- make -j 64
```

## Usage
Suppose the current working directory is 'KEAG'. We provide three examples demonstrating: (1) training data collection, (2) model training, and (3) model application in an R-tree.

### (1) Collect training data 

To collect training data for the Split operation on the guassian-2M dataset, execute the following scripts:
```bash
- cd libspatialindex/build/test
- ./AccessGT -d guassian -s 2 -gt split
```
Then, the training data is collected and saved to the dataset directory.

### (2) Train the neural network in KEAG

Suppose you want to train the Split-version KEAG using the training data just collected. Execute the following scripts.
```bash
- cd src
- python run.py --task split --train_model True --data guassian --data_size 2
```
The trained KEAG is exported as a TorchScript module and saved to the directory 'KEAG/ckpt'.

### (3) Apply the trained model in an R-tree

Suppose you want to use the trained model to select candidates for R-tree Split operations. Execute the following scripts.
```bash
- cd libspatialindex/build/test
- ./Eval -d guassian -s 2 -mt split -ua false
```
Alternatively, you can directly use the TorchScript module we provided previously. Execute the following scripts.
```bash
- cd libspatialindex/build/test
- ./Eval -d guassian -s 2 -mt split -ua true
```
