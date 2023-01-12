# A-HAIR
The human intelligence -> game -> artificial intelligence paradigm application on molecular simulation using fighting game. The game Samurai Shodown II has been used, which has been purchased for play on Steam using the Steam ID 278437426.

## 1 Sample
Sampling of the untuned model data, concerning mainly about the model weight and the loss. This procedure produces a loss surface of the model parameters, unveiling to us the local difficulties the model might trapped be in now.

The model is written in the script ***model_E_rotate_molecule_decay.py***, and can be trained with ***train.py***. The script ***collect_weight.py*** is used to get information about the loss surface of the model.

## 2 Train and Fight
Train the AI opponents using the MPO method, and fight with it to gain game data. During the fighing process, the loss surface from the first step is mapped to the AI opponent to show to the players similar scenarios for them to solve.

The AI opponent can be trained and be fought with using the script ***train_or_fight_ukyo.py*** by altering the argument --train. ***readGameData.py*** is used to convert the game data into numerical arrays. ***weight_analyze.py*** is used to analyze the collected game data and coach the AI opponent during the fight.

The game data is collected using screenshots and read using yolo (Jocher, G. (2020). YOLOv5 by Ultralytics (Version 7.0) [Computer software]. https://doi.org/10.5281/zenodo.3908559). Run ***recordJoysticks.py*** before fighting the AI opponent or simpy playing the game, the printed info in the command line will tell you what to do. Be sure to play Samurai Shodown II from the Samurai Shodown NEOGEO Collection on Steam.

The settings of the game are "WINDOW", "960x540", "ON", "HIGH".
The settings of Samurai Shodown II are Difficulty "6", "FULL", "OFF", and "SS2 B".
The controls for 1P are U, I, J, K, O, L with directions using W, A, S, D.
The controls for 2P are R, T, F, G, Y, H with directions using the direction keys.
The "Start" for 1P and 2P are F1, F2, respectively.

For training, the player plays Haohmaru as 1P and the opponent plays Ukyo as 2P.
For the fighting, the AI plays Haohmaru as 2P and the player plays Ukyo as 1P.

## 3 Tune
The game data collected from the last step is converted into some BO optimizers to assist the Adam optimizer usually used for the neural networks.

***GameUtilFuncTrain.py*** should be run first to train the utility function using human game data. Then ***tune.py*** applies this utility function summarized from human game data to modify the Adam optimizer in a Bayesian way and tune the model.

## 4 Other Models
The GOODLE model (https://github.com/XI-Lab/CA) was modified to finite version here for molecules and tuned.
Other models can also be tuned with our method. SchNet (https://github.com/atomistic-machine-learning/schnetpack), TorchANI (https://github.com/aiqm/torchani), and SpookyNet (https://github.com/OUnke/SpookyNet) are shown here.

## 5 Requirements
The required packages are as shown in the file env_info.txt.
