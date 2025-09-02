# Pre - volumereco for GRAIN
## Introduction
DUNE is a new generation long baseline neutrino experiment with a wide spectrum of scientific goals like the determination of neutrino mass ordering, the study of CP symmetry violation in the leptonic sector, the observation of supernova neutrinos and physics beyond the Standard Model. It will consist of a Near Detector at 547 m from the neutrino source and a Far Detector placed ~1300 km apart. One component of the Near Detector complex is the experimental apparatus called SAND. One component of SAND is the GRAIN (GRanular Argon for Interactions of Neutrinos) sub-detector, a new liquid Argon detector that aims to perform imaging of neutrino interactions using scintillation light, providing vertexing and tracking information.

### 3D imaging with scintillation light
An innovative cryogenic light readout system for GRAIN consists in a matrix of SiPMs with the optic realized on coded aperture masks (a grid of alternating opaque material and holes). The reconstruction algorithm, based on Maximum Likelihood Expectation-Maximization, combines the views of about 60 cameras providing a three-dimensional map of the energy deposited by charged particles. This iterative approach represents a significant computational challenge because it requires optimized use on a computing system with multiple GPUs.

https://github.com/user-attachments/assets/22fd6985-2855-4186-83be-e50822842c8a

### Goal of pre-volumereco
The goal of this project is to give an initial rough estimate of the expected 3D energy deposit, to be used as a starting point for the Maximum Likelihood Expectation-Maximization algorithm (instead of starting with a flat distribution). This is expected to make the algorithm converge faster, saving (GPU) time.

