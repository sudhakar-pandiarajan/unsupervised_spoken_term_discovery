'''
	Author: Sudhakar. P 
	Updated on : April 30, 2024
'''
import sys
import os
from os import path
from tqdm import tqdm
import glob
import numpy as np
import matplotlib
import pandas as pd
import audiofile
import argparse
import soundfile as sf
import torch
import torchaudio
from data import Wav2Mel
#from rasta import *
class RASTAPLP: 
	'''
		Extract the RASTA PLP spectrogram
	'''
	def generate_rasta_plp_spec(self,wav_path, des_path): 
		print("Processing RASTA-PLP feature extraction...")
		for fname in tqdm(sorted(glob.glob(wav_path))): 
			file_name= path.splitext(path.split(fname)[-1])[0]
			y, sr = audiofile.read(fname)
			cepstra, spectra = rastaplp(y, sr, win_time=0.02,hop_time=0.02)
			f_path = path.join(des_path, file_name)
			np.save(f_path,spectra)
		print("RASTA feats done!")

class MelSpec: 
	'''
		This class implements the speaker independent mel-spectrogram representation using CNN
	'''
	def generate_speaker_norm_melspec(self, wav_path, des_path, model_path, tgt_spkr_path): 
		
		print("Processing MelSpec Normalised feature extraction...")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = torch.jit.load(model_path).to(device)
		for fname in tqdm(sorted(glob.glob(wav_path))): 
			file_name= path.splitext(path.split(fname)[-1])[0]
			bnf_feats = self.get_bottleneck(model,fname, tgt_spkr_path)
			f_path = path.join(des_path, file_name)
			np.save(f_path, bnf_feats)
			
	def get_bottleneck(self, model, doc_path, tar_spkr_path): 
		'''
			returns the target speaker char of the src speaker bottleneck mel-spec
		'''
		wav2mel = Wav2Mel()
		
		kw, kw_sr = torchaudio.load(doc_path)
		ref, ref_sr = torchaudio.load(tar_spkr_path)
		
		device = "cuda" if torch.cuda.is_available() else "cpu"
		kw_in = wav2mel(kw, kw_sr)[None, :].to(device)
		ref_in = wav2mel(ref, ref_sr)[None, :].to(device)
		cvt_kw_ref = model.inference(kw_in, ref_in) # spoken content of the doc is mapped to the tar spkear
		
		bnf_kws = cvt_kw_ref.squeeze(0).data.T.numpy()
		
		'''
		wav2mel = Wav2Mel()
		src, src_sr = torchaudio.load(doc_path)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		src_feats = wav2mel(src, src_sr)[None, :].to(device) # speech doc
		
		kws, kws_sr = torchaudio.load(tar_spkr_path)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		kws_feats = wav2mel(kws, kws_sr)[None, :].to(device) # query doc
		
		cvt_kws = model.inference(src_feats, kws_feats)
		bnf_kws = cvt_kws.squeeze(0).data.T.numpy()
		'''
		return bnf_kws	
	
