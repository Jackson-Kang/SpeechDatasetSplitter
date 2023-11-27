# SpeechDatasetSplitter
A simple waveform segmentator using OpenAI's Whisper


# Introduction
Recently proposed deep-learning based speech processing models require batch processing due to limited memory space. To train the models, generally, we need to segment a long waveform into utterance-level segments. In the wild, however, most of the speech dataset collected from record studio or web-scrapping contains long waveform which consists of multiple utterances. To automate the segmentation process, we propose SpeechDatasetSplitter. The proposed SpeechDatasetSplitter is built on top of the [OpenAI's Whisper](https://github.com/openai/whisper/tree/main), and capable to sophisticatedly segment long-waveform into multiple utterances. Moreover, automated annotation and validation are also possible for efficient and effective data preprocessing.


# Installation
* install pre-requisite via following command.

```
pip install -r requirements.txt
```

# How-to-use




# Contacts

If any question, please email to mskang1478@gmail.com.

