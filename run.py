"""
	Program Author: Minsu Kang (2023.11.27.)

	Description)
		A simple waveform segmentator using OpenAI's Whipser.

		This script divides long waveform utterances into multiple segments by detecting punctuation marks such as ".", "?" and "!".

"""


#	[TODO] Pseudo Codes)
#		(pre-requsite) testcase generator
#		
#		(1) divide long-waveform utterances into multiple short segments by algorithmic silence detection
#
#		(2) recognize short segments using OpenAI's Whisper.
#
#		(3) merge recognized segments with detected punctuation marks.
#

from config import Arguments as args

from utils import load_waveform, get_filelist, merge_waveform_segments, save_waveform, get_path, get_whisper, split_long_waveform, create_dir
from tqdm import tqdm


def run():

	device = "cuda:{}".format(args.DEVICE)

	# STEP 01 - Get waveform from disk or generate testcase

	print("\t [LOG] STEP 01 - Load waveform...")
	if args.IS_TEST:
		waveform_segment_filelist = get_filelist(args.TEST_DATASET_PATH, file_format="wav")
		waveform_segments_list = [load_waveform(waveform_segment_filepath, args.SAMPLE_RATE) 
						for waveform_segment_filepath in tqdm(waveform_segment_filelist, total=len(waveform_segment_filelist))]

		wav = merge_waveform_segments(waveform_segments_list)	
		savepath = get_path(args.DATASET_PATH, "long_waveform_kss.wav")
		save_waveform(savepath, wav, args.SAMPLE_RATE)
		wavepath = savepath

	else:
		# [TODO] Expand to multispeaker cases
		waveform_segment_filelist = get_filelist(args.TEST_DATASET_PATH, file_format="*.wav")
		waveform_segments_list = [load_waveform(waveform_segment_filepath, args.SAMPLE_RATE)]
	print("\t\t Done..!\n\n")


	print("\t [LOG] STEP 02 - Split waveform into segments")
	print("\t\t This process make take several times... Please be patient..")
	model = get_whisper(model_name=args.WHISPER_MODEL_TYPE, device=device)
	result = model.transcribe(wavepath)	
	print("\t\t Done..!\n\n")


	print("\t [LOG] STEP 03 - Split segments")	
	create_dir(args.RESULT_SAVEDIR)
	splitted_waveform_list = split_long_waveform(wav, result['segments'])
	[save_waveform("{}/{:0>4}.wav".format(args.RESULT_SAVEDIR, idx), splitted_waveform, sample_rate=args.SAMPLE_RATE) for idx, splitted_waveform in tqdm(enumerate(splitted_waveform_list), total=len(splitted_waveform_list))]
	print("\t\t Done..!\n\n")



if __name__ == "__main__":

	run()

