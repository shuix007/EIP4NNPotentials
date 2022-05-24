# Injecting Domain Knowledge from Empirical Interatomic Potentials to Neural Networks for Predicting Material Properties

This is the code for our submission Injecting Domain Knowledge from Empirical Interatomic Potentials to Neural Networks for Predicting Material Properties to NeurIPS 2022.

## Requirement

Our implementation is based on Deep Graph Library (DGL) and PyTorch. To run this code, you need

```
ase=3.22.1
dscribe=1.2.1
kim-api=2.2.1
kimpy=2.0.0
openkim-models=2021.01.28
dgl=0.7.2
pytorch=1.7.1
```

## Reproducing our experimental results on the ANI-Al dataset
 
Download the Al-data.tgz data file from https://github.com/atomistic-ml/ani-al/blob/master/data/Al-data.tgz to the ```RawData/ANI-Al``` folder. We provide bash scripts in the ```Scripts``` folder to preprocess the raw dataset and to run experiments.

To process the dataset into the extended xyz form, compute EIP energies using KIM, compute SOAP descriptors for the configurations, and construct graphs for the configurations, change to the `Scripts` directory and run the following:
```
bash preprocess.sh
```

To reproduce our results in Table 1, run the following from the `Scripts` directory:
```
bash experiments.sh
```
