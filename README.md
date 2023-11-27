# SpeechDatasetSplitter
A simple waveform segmentator using OpenAI's Whisper


# Introduction
Recently proposed deep-learning based speech processing models require batch processing due to limited memory space. To train the models, generally, we segment a long waveform into utterance-level segments. In the wild, however, most of the speech dataset collected from record studio or web-scrapping contains long waveform which consists of multiple utterances, and makes inappropriate to train them. Manual segmentation and annotation are needed in some cases, but 
time-inefficient. 

We propose SpeechDatasetSplitter to automate the segmentation and annotation process. The proposed SpeechDatasetSplitter is built on top of the [OpenAI's Whisper](https://github.com/openai/whisper/tree/main), and capable to sophisticatedly segment long-waveform into multiple utterances. Moreover, automated annotation and validation are also possible for efficient and effective data preprocessing.


# Installation
* strongly recommend to run under [Anaconda](https://www.anaconda.com/download) environment
* install pre-requisite via following command.

```
pip install -r requirements.txt
```

# How-to-use
* upload waveform in the ```samples``` directory.
* run script(```run.py```) via following command
```
python run.py
```
* check the segmented waveforms in ```results``` directory   
* you may check the error report named ```result.csv```.


# Contacts

If any question, please email to mskang1478@gmail.com.

