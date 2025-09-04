# pre-volumereco for GRAIN
## Introduction
DUNE is a new generation long baseline neutrino experiment with a wide spectrum of scientific goals like the determination of neutrino mass ordering, the study of CP symmetry violation in the leptonic sector, the observation of supernova neutrinos and physics beyond the Standard Model. It will consist of a Near Detector at 547 m from the neutrino source and a Far Detector placed ~1300 km apart. One component of the Near Detector complex is the experimental apparatus called SAND. One component of SAND is the GRAIN (GRanular Argon for Interactions of Neutrinos) sub-detector, a new liquid Argon detector that aims to perform imaging of neutrino interactions using scintillation light, providing vertexing and tracking information.

<p align="center">
  <img width="50%" height="50%" alt="grain" src="https://github.com/user-attachments/assets/03713bb0-eb98-4b99-a5bc-7bb80c267fba" />
</p>

### 3D imaging with scintillation light
An innovative cryogenic light readout system for GRAIN consists in a matrix of SiPMs with the optic realized on coded aperture masks (a grid of alternating opaque material and holes). The reconstruction algorithm, based on Maximum Likelihood Expectation-Maximization, combines the views of about 60 cameras providing a three-dimensional map of the energy deposited by charged particles. This iterative approach represents a significant computational challenge because it requires optimized use on a computing system with multiple GPUs.

https://github.com/user-attachments/assets/22fd6985-2855-4186-83be-e50822842c8a

## Goal of pre-volumereco
The goal of this project is to give an initial rough estimate of the expected 3D energy deposit, to be used as a starting point for the Maximum Likelihood Expectation-Maximization algorithm (instead of starting with a flat distribution). This is expected to make the algorithm converge faster, saving (GPU) time.

### Project structure
- <code>prepare_input_data.py</code> converts output data from GRAIN Monte Carlo simulations into a ML-friendly data format (numpy arrays saved as <code>.h5</code> file). Modules from [sand-optical](https://baltig.infn.it/dune/sand-optical/tools) are used to read Monte Carlo data.
- <code>data/lightweight_dataset_20cm.h5</code> is a ML-ready dataset provided as an example.
- <code>train.ipynb</code> is a notebook used to explore input data, train a deep neural network optimizing hyperparameters with [OPTUNA](https://optuna.org), save the model and show some predictions.
- <code>saved_models/pre_volumereco_optuna_20cm.keras</code> is an already trained model that can be used as an example, skipping the training step.

### Model evaluation
Compare reconstructions starting from a flat voxel score distribution and starting from the initial prediction of the model. Given a set number of iterations of the reconstruction algorithm, count how many iterations are necessary to reach the 95% maximum likelihood (which should converge iteration after iteration). The expectation is that starting from the initial prediction of the model lowers the number of iterations required.
