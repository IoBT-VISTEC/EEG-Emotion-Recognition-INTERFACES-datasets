# EEG-Emotion overview

Steps:
    - Data Preprocessing -> Feature Extraction -> Classifier (Emotion recognition)
    
Note:
    - Import data from data-lake (https://www.dropbox.com/sh/l0fxvbcvf15vnv1/AACMiqjFOMse6ODftzU7AKMxa?dl=0) to folder ./data/


## Database Description
After data acquisition, The data were processed and extracted features with method at previous subsection. Emotion database is available in data lake. The structure and file description can be described as follows:
    * • Task 2-5 Emotion/
    * • EEG/
    * • feature extracted/
· EEG ICA.npy: Power Spectral Density of each frequency band and channel as Table 4.1 with ICA method in shape (N.subjects* N.clips, N.channels, N.freq bands, 1) = (645,8, 4, 1)
· EEG no ICA.npy: Power Spectral Density of each frequency band and channel as Table 4.1 with out ICA method in shape (N.subjects* N.clips, N.channels, N.freq bands, 1) = (645, 8, 4, 1)
    * • preprocessed/
· EEG ICA.npy: EEG signal with ICA method in shape (N.subjects* N.clips, N.channels, N.freq bands, N.sampling points (56sec)) = (645, 8, 4, 14000)
· EEG no ICA.npy: EEG signals with out ICA method in shape (N.subjects* N.clips, N.channels, N.freq bands, N.sampling points (56sec)) = (645, 8, 4, 14000)
    * • raw/
· EEG.npy: raw EEG signals(µV) with sampling rate 250 Hz recorded from OpenBCI in shape (N.subjects, N.clips, N.channels, N.sampling points(56sec)) = (43, 15, 8, 14000)
    * • EEG/
    * • feature extracted/
· BVP.npy: Data from photoplethysmography after preprocessing as Table 4.1 in shape (N.subject, N.clip, N.of features)=(43, 15, 13)
· EDA.npy: Data from the electrodermal activity sensor after preprocessing as Table 4.1 in shape (N.subject, N. clip, N.features) = (43, 15, 21)
· TEMP.npy: Data from Data from temperature sensor after preprocessing as Table 4.1 in shape (N.subject, N.clip, N.features) = (43, 15, 4)
    * • raw/
· ACC.npy: Data from 3-axis accelerometer sensor with sampling rate 32 Hz recorded from Empatica E4 in shape (N.subject, N.clip, N. x, y, and z axis, N.sampling points (56 sec)) = (43, 15, 3, 1792)
· BVP.npy: Data from photoplethysmography with sampling rate 64 Hz recorded from Empatica E4 in shape (N.subject, N. clip, N.sampling points (56 sec)) = (43, 15, 3584)
· EDA.npy: Data from the electrodermal activity sensor expressed as microsiemens (µS) with sampling rate 4 Hz recorded from Empatica E4 in shape (N.subject, N.clip, N.sampling points (56 sec)) = (43, 15, 224)
· HR.npy: Data from heart rate with sampling rate 1 Hz recorded from Empatica E4 in shape (N.subject, N.clip, N.sampling points (56 sec)) = (43, 15, 56)
· HRV.npy: Heart rate variability recorded from Empatica E4 in shape (N.subject, N.clip) = (43, 15)
· IBI.npy: Inter-beat interval recorded from Empatica E4 in shape (N.subject, N.clip) = (43, 15)
· TEMP.npy: Data from Data from temperature sensor (°C) with sampling rate 4 Hz recorded from Empatica E4 in shape (N.subject, N.clip, N.sampling points (56 sec)) = (43, 15, 224)
    * • score/
    
    * • label/
· arousal.npy: Labeling by of arousal score (0:low or 1:high).
· excite.npy: Labeling by of excite score (0:low or 1:high).
· fear.npy: Labeling of fear score (0:low or 1:high).
· happy.npy: Labeling of happy score (0:low or 1:high).
· rating.npy: Labeling of rating score (0:low or 1:high).
· valence.npy: Labeling of valence score (0:low or 1:high).
    * • raw/
· Raw.npy: Self emotional score of all participants in shape (43, 15, 1) = ( N.subject, No. of clip, emotional score) in each emotion (happy, fear, excite, arousal, valence, rating).
    * • clip/
· All video clips which were played to participants.
    * • Clip list.csv : Name of clips which were played for each participant. (15 clips/person)
