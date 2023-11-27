from glob import glob
from scipy.io.wavfile import write
import librosa, os
import numpy as np
import scipy
import soundfile as sf
import whisper


MAX_WAV_VALUE = 32768

def load_waveform(path, sample_rate):

	wav, sr = librosa.load(path, sr=sample_rate)

	assert sr == sample_rate, "[ERROR] Sample rate differs! (config - {} vs. audio - {})".format(sample_rate, sr)

	return wav


def save_waveform(savepath, waveform, sample_rate):
	sf.write(savepath, waveform, sample_rate, 'PCM_16')


def get_filelist(dirname, file_format):
	filepath = get_path(dirname, "*.{}".format(file_format))
	return list(glob(filepath))


def get_path(*args):
	return os.path.join('', *args)


def create_dir(*args):
        path = get_path(*args)
        if not os.path.exists(path):
                os.mkdir(path)
        return path

def merge_waveform_segments(waveform_segment_list):

	# input: list of 1D numpy array (each array has waveform segment)
	# output: 1D numpy array that contains waveform signal

	long_waveform = []
	for waveform_segment in waveform_segment_list:
		long_waveform.extend(waveform_segment.tolist())
	
	return np.array(long_waveform).astype('float')


def split_long_waveform(wav, segment_list, sample_rate=22050):

	splitted_waveform = []
	for segment in segment_list:
		start_sec, end_sec = segment['start'], segment['end']
		start_idx, end_idx = round(start_sec * sample_rate), round(end_sec * sample_rate)
		splitted_waveform.append(wav[start_idx: end_idx])
	return splitted_waveform


def get_whisper(model_name, device):
	model = whisper.load_model(model_name).to(device)
	return model




