# pre-volumereco for GRAIN
## Introduction
DUNE is a next-generation long-baseline neutrino experiment aiming to determine the neutrino mass ordering, study CP violation in the leptonic sector, observe supernova neutrinos, and search for physics beyond the Standard Model. It will feature a Near Detector 547 m from the source and a Far Detector ~1300 km away. Within the Near Detector, the SAND apparatus includes GRAIN (GRanular Argon for Interactions of Neutrinos), a novel liquid argon detector designed to image neutrino interactions via scintillation light, providing vertexing and tracking.

<p align="center">
  <img width="50%" height="50%" alt="sand" src="https://github.com/user-attachments/assets/03713bb0-eb98-4b99-a5bc-7bb80c267fba" />
  <img width="30%" height="30%" alt="grain" src="https://github.com/user-attachments/assets/7cbab096-ac4a-4e1a-a12b-d1754ba9c02f" />
</p>

### 3D imaging with scintillation light
An innovative cryogenic light readout system for GRAIN consists in a matrix of SiPMs with the optic realized on coded aperture masks (a grid of alternating opaque material and holes). The reconstruction algorithm, based on Maximum Likelihood Expectation-Maximization, combines the views of 60 cameras providing a three-dimensional map of the energy deposited by charged particles. This iterative approach represents a significant computational challenge because it requires optimized use on a computing system with multiple GPUs.

<p align="center">
  <img width="35%" height="35%" src="https://github.com/user-attachments/assets/a307cbaf-3ee5-4632-a768-266bebba43c9" alt="animated_mutrack" />
</p>

## Goal of pre-volumereco
The goal of this project is to give an initial rough estimate of the expected 3D energy deposit, to be used as a starting point for the Maximum Likelihood Expectation-Maximization algorithm (instead of starting with a flat distribution). This is expected to make the algorithm converge faster, saving (GPU) time.

### Project structure
- <code>prepare_input_data.py</code> converts output data from GRAIN Monte Carlo simulations into a ML-friendly data format (numpy arrays saved as <code>.h5</code> file). Modules from [sand-optical](https://baltig.infn.it/dune/sand-optical/tools) are used to read Monte Carlo data.
- <code>data/lightweight_dataset_20cm.h5</code> is a ML-ready dataset provided as an example.
- <code>train.ipynb</code> is a notebook used to explore input data, train a deep neural network optimizing hyperparameters with [OPTUNA](https://optuna.org), save the model and show some predictions.
- <code>saved_models/pre_volumereco_optuna_20cm.keras</code> is an already trained model that can be used as an example, skipping the training step.

### Dataset and predicitons
The event features are the average hit times in each one of camera, the event thruths are the voxelized energy deposits. Currently, given the limited available dataset it is necessary to consider large 20x20x20 $$\text{cm}^{3}$$ voxels.

![features_vs_truth](https://github.com/user-attachments/assets/2fa61bd9-5ee0-4e89-98a9-177e9cc3a877)
![prediction_vs_truth](https://github.com/user-attachments/assets/4cd3df5a-d8e4-4520-87a7-b08d587b6793)


### Model evaluation
Compare reconstructions starting from a flat voxel score distribution and starting from the initial prediction of the model. Given a set number of iterations of the reconstruction algorithm, count how many iterations are necessary to reach the 99% maximum likelihood (which should converge iteration after iteration). The expectation is that starting from the initial prediction of the model lowers the number of iterations required.

<p align="center">
  <img width="45%" height="45%" alt="likelihood" src="https://github.com/user-attachments/assets/da34aca9-52e9-4e90-b79a-9282ec36ae1f" />
</p>
