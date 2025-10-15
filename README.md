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
The goal of this project is to provide a prior of the expected three-dimensional energy deposition to serve as a seed for the MLEM algorithm, rather than a uniform distribution. This improves convergence and reduces GPU load.

### Project structure
- <code>prepare_input_data.py</code> converts output data from GRAIN Monte Carlo simulations into a ML-friendly data format (numpy arrays saved as <code>.h5</code> file). Modules from [sand-optical](https://baltig.infn.it/dune/sand-optical/tools) are used to read Monte Carlo data.
- <code>data/lightweight_dataset_20cm.h5</code> is a ML-ready dataset provided as an example.
- <code>train.ipynb</code> is a notebook used to explore input data, train a deep neural network optimizing hyperparameters with [OPTUNA](https://optuna.org), save the model and show some predictions.
- <code>saved_models/pre_volumereco_optuna_20cm.keras</code> is an already trained model that can be used as an example, skipping the training step.

### Dataset and predicitons
The event features are the average hit times in each one of camera, the event thruths are the voxelized energy deposits. Currently, given the limited available dataset it is necessary to consider large 20x20x20 $$\text{cm}^{3}$$ voxels.

![features_vs_truth](https://github.com/user-attachments/assets/0c02b59d-2916-4dfc-991f-1e89e08fffa3)
<p align="center">
  <img width="768" height="344" alt="pred_vs_reco" src="https://github.com/user-attachments/assets/77f271ee-c234-401a-a9a1-574e1d5c382f" />
</p>

### Model evaluation
The trained model was evaluated on 600 events as a prior for MLEM reconstruction, showing that likelihood convergence is reached approximately 20 iterations earlier compared to a uniform prior. Convergence is defined as a log likelihood change <50 between iterations.

<p align="center">
  <img width="829" height="295" alt="likelihood" src="https://github.com/user-attachments/assets/f37c73c3-450f-42e7-8f6e-a411223872fd" />
</p>
