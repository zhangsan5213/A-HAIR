# HIgameAI

## 1 Sample
### Sampling of the untuned model data, concerning mainly about the model weight and the loss. This procedure produces a loss surface of the model parameters, unveiling to us the local difficulties the model might trapped be in now.

## 2 Train and Fight
### Train the AI opponents using the MPO method, and fight with it to gain game data. During the fighing process, the loss surface from the first step is mapped to the AI opponent to show to the players similar scenarios for them to solve.

## 3 Tune
### The game data collected from the last step is converted into some BO optimizers to assist the Adam optimizer usually used for the neural networks.

## 4 Other Models
### The GOODLE model (Lin, H., Ye, S., & Zhu, X. (2022). Geometry Orbital of Deep Learning (GOODLE): A uniform carbon potential. Carbon, 186, 313-319.) was modified to finite version here for molecules and tuned.
### Other models can also be tuned with our method. SchNet (https://github.com/atomistic-machine-learning/schnetpack), TorchANI (https://github.com/aiqm/torchani), and SpookyNet (https://github.com/OUnke/SpookyNet) are shown here.

## Requirements
### The required packages are as shown in the file env_info.txt.
