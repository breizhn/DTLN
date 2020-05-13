# Dual-signal Transformation LSTM Network
Implementation of the stacked dual-signal transformation LSTM network (DTLN) for real-time noise supression. \
\
This model was handed in to the deep noise suppression challenge ([DNS-Challenge](https://github.com/microsoft/DNS-Challenge)) of Interspeech 2020. 
\
\
This approach combines a short-time Fourier transform (STFT) and a learned analysis and synthesis basis in a stacked-network approach with less than one million parameters. The model was trained on 500h of noisy speech provided by the challenge organizers. The network is capable of real-time processing (one frame in, one frame out) and reaches competitive results.
Combining these two types of signal transformations enables the DTLN to robustly extract information from magnitude spectra and incorporate phase information from the learned feature basis. The method shows state-of-the-art performance and outperforms the DNS-Challenge baseline by 0.24 points absolute in terms of the mean opinion score (MOS).
\
\
Author: Nils L. Westhausen ([Communication Acoustics](https://uol.de/en/kommunikationsakustik) , Carl von Ossietzky University, Oldenburg, Germany)
