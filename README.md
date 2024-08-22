
# Unsupervised Spoken Term Discovery using Recurring Speech Pattern

Objective: This research work demonstrates the spoken term discovery task in the zero-resource constraint by directly computing the spoken term matches from the acoustic feature representation itself.

The objective was achieved in two stages: 
1. Acoustic Feature representation
2. Spoken term match computation (discovery)

python implementation of both stages are available in the AcousticFeats.py and Discovery.py files. 


## Acoustic Feature representation
The objective of the acoustic feature representation is to identify the speaker independent spoken content representation from the speech signal. To achieve this, we analysed Mel-Spectrogram and RASTA-PLP Spectrogram representation. 

The implementation of the acoustic feature extraction was programmed in the AcousticFeats.py

## Spoken Term Discovery
Spoken term discovery module aims to compute the pattern matches from the acoustic feature representation and detect the spoken term match. 
The proposed recurring pattern match discovery was demonstrated in the Discovery.py file.

## Dependency
Following python packages are required for execution 
1. numpy 
2. matplotlib
3. pandas 
4. audiofile
5. torch
6. torchaudio
7. tqdm 
8. pickle
9. scipy
