# EEG-Emotion overview

Steps:
    - Data Preprocessing -> Feature Extraction -> Classifier (Emotion recognition)
    
Note:
    - Import data from data-lake (https://www.dropbox.com/sh/l0fxvbcvf15vnv1/AACMiqjFOMse6ODftzU7AKMxa?dl=0) to folder ./data/


## Database Description
After data acquisition, The data were processed and extracted features with method at previous subsection. Emotion database is available in data lake. The structure and file description can be described as follows:


    • Task 2-5 Emotion/
        • EEG/
            • feature extracted/
                · EEG ICA.npy: Power Spectral Density of each frequency band and channel as Table 4.1 with ICA method in shape (N.subjects* N.clips, N.channels, N.freq bands, 1) = (645,8, 4, 1)
                · EEG no ICA.npy: Power Spectral Density of each frequency band and channel as Table 4.1 with out ICA method in shape (N.subjects* N.clips, N.channels, N.freq bands, 1) = (645, 8, 4, 1)
            • preprocessed/
                · EEG ICA.npy: EEG signal with ICA method in shape (N.subjects* N.clips, N.channels, N.freq bands, N.sampling points (56sec)) = (645, 8, 4, 14000)
                · EEG no ICA.npy: EEG signals with out ICA method in shape (N.subjects* N.clips, N.channels, N.freq bands, N.sampling points (56sec)) = (645, 8, 4, 14000)
            • raw/
                · EEG.npy: raw EEG signals(µV) with sampling rate 250 Hz recorded from OpenBCI in shape (N.subjects, N.clips, N.channels, N.sampling points(56sec)) = (43, 15, 8, 14000)
        • E4/
            • feature extracted/
                · BVP.npy: Data from photoplethysmography after preprocessing as Table 4.1 in shape (N.subject, N.clip, N.of features)=(43, 15, 13)
                · EDA.npy: Data from the electrodermal activity sensor after preprocessing as Table 4.1 in shape (N.subject, N. clip, N.features) = (43, 15, 21)
                · TEMP.npy: Data from Data from temperature sensor after preprocessing as Table 4.1 in shape (N.subject, N.clip, N.features) = (43, 15, 4)
            • raw/
                · ACC.npy: Data from 3-axis accelerometer sensor with sampling rate 32 Hz recorded from Empatica E4 in shape (N.subject, N.clip, N. x, y, and z axis, N.sampling points (56 sec)) = (43, 15, 3, 1792)
                · BVP.npy: Data from photoplethysmography with sampling rate 64 Hz recorded from Empatica E4 in shape (N.subject, N. clip, N.sampling points (56 sec)) = (43, 15, 3584)
                · EDA.npy: Data from the electrodermal activity sensor expressed as microsiemens (µS) with sampling rate 4 Hz recorded from Empatica E4 in shape (N.subject, N.clip, N.sampling points (56 sec)) = (43, 15, 224)
                · HR.npy: Data from heart rate with sampling rate 1 Hz recorded from Empatica E4 in shape (N.subject, N.clip, N.sampling points (56 sec)) = (43, 15, 56)
                · HRV.npy: Heart rate variability recorded from Empatica E4 in shape (N.subject, N.clip) = (43, 15)
                · IBI.npy: Inter-beat interval recorded from Empatica E4 in shape (N.subject, N.clip) = (43, 15)
                · TEMP.npy: Data from Data from temperature sensor (°C) with sampling rate 4 Hz recorded from Empatica E4 in shape (N.subject, N.clip, N.sampling points (56 sec)) = (43, 15, 224)
        • score/
    
        • label/
                · arousal.npy: Labeling by of arousal score (0:low or 1:high).
                · excite.npy: Labeling by of excite score (0:low or 1:high).
                · fear.npy: Labeling of fear score (0:low or 1:high).
                · happy.npy: Labeling of happy score (0:low or 1:high).
                · rating.npy: Labeling of rating score (0:low or 1:high).
                · valence.npy: Labeling of valence score (0:low or 1:high).
        • raw/
                · Raw.npy: Self emotional score of all participants in shape (43, 15, 1) = ( N.subject, No. of clip, emotional score) in each emotion (happy, fear, excite, arousal, valence, rating).
        • clip/
                · All video clips which were played to participants.
        • Clip list.csv : Name of clips which were played for each participant. (15 clips/person)
Manual
 =
## Contents
1.	Pre-installation
    • Set up python libraries: numpy, scipy, sklearn, mne, pandas, and matplotlib
    • Create a directory named data at emotion-monitoring-system/data
    • Copy data from data-lake to the above directory
2. Pre-processing data
    • Go to ./src
    • Open and run all cells in EEGPreprocessing.ipynb
    • Answer the question ”Do you want to re-run all? (y/n):”
        – If this is the first time of preprocessing the data, type y.
        – Otherwise, type y if you want to re-run all again or n if you want to continue from the latest pre-processed signal.
    • The program will perform preprocessing to each sample including
        – Independent Component Analysis (ICA): In this step, it allows experts to specify which components should be removed from
    the EEG signals.
        – Common Average Reference (CAR)
        – Bandpass filter to sub-frequency bands including
        – Reshape data to (number of samples per subject * number of subjects, number of channels, number of sub-frequency bands, number of sampling points) = (645, 8, 4, 14000)
    • The program automatically saves all data into data/EEG/preprocessed/EEG_ICA.npy
3.Feature Extraction
    • EEG
        – Go to ./src
        – Open and run all cells in EEGFeatureExtraction.ipynb
        – The software automatically
            * Calculates Power Spectral Density (PSD) of each sub frequency band.
            * Saves into data/feature_extracted/EEG.npy
    • Body signals
        – Go to ./src
        – Open and run all cells in E4_Extract_Feature.ipynb
        – The software automatically
            * Calculate all features from E4 (Empatica)
            * Saves EDA.npy, TEMP.npy, and BVP.npy into data/E4/feature_extracted/



### Our Paper
When using (any part) of this dataset, please cite [our paper](https://ieeexplore.ieee.org/document/8762012)
