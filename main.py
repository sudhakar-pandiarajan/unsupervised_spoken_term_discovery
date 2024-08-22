'''
	Author: Sudhakar. P 
	Updated on : April 30, 2024
'''
from AcousticFeats import *
from Discovery import *

def get_target_speaker(audio_path,lang):
	tgt_spkr = {"ta":"000020204", "te":"000020261","gu":"000010005"} # choose target speaker for speaker normalisation
	tgt_spkr_path = path.join(audio_path,"corpus",lang,"{}.wav".format(tgt_spkr[lang]))
	return tgt_spkr_path

def generate_acousticFeats(app_path, wav_path, feats_type="RASTA-PLP", lang="ta"): 
	aud_path = path.join(wav_path,"corpus",lang,"*.wav")
	des_path = path.join(app_path,"AcousticFeats",feats_type, lang)
	print("------- Acoustic feature extraction ---------------")
	print("Current Language :", lang)
	print("Feats type: ",feats_type)
	if(feats_type=="RASTA-PLP"):
		rasta = RASTAPLP()
		rasta.generate_rasta_plp_spec(aud_path, des_path)
	elif(feats_type=="Mel-Spec"): 
		tgt_spkr = get_target_speaker(app_path,lang)
		print("Normalising audio to the tgt spkr ", tgt_spkr)
		model_path =path.join(app_path, "Models","Mel-spec-norm", lang, "model-10000-{}.ckpt".format(lang))
		melspec = MelSpec()
		melspec.generate_speaker_norm_melspec(aud_path,des_path, model_path, tgt_spkr) 

def spoken_term_discovery(base_path, lang, feats_type):
	feats_path = path.join(base_path, "AcousticFeats",feats_type,lang, "*.npy")
	result_path = path.join(base_path, "Results", lang) 
	dis = Discovery()
	print("-----------Spoken Term Discovery started...-----------")
	print("Language :", lang)
	print("Feats type :", feats_type)
	dis.compute_word_pair_match(feats_path, result_path, sim_thre=0.99, depth_thre=11)

if __name__=="__main__": 
	
	print(" Spoken Term discovery Task...")
	# initial parameter configuration 
	app_path = "/home/sudhakar-gpu/Sudhakar/PhD/Framework/Speech/Premi_invited_DCE" # Current Application path
	audio_path = "/home/sudhakar-gpu/Sudhakar/PhD/Framework/Speech/Premi_invited_DCE" # Speech corpus path
	lang = ['ta','te', 'gu', 'hi'] # select any one language
	feats = ["RASTA-PLP", "Mel-Spec"] # RASTA --> RASTA PLP Spectrogram, Mel-Spec --> speaker normalised Mel-spectrogram representation

	#----------------- Feature extraction --------------
	# step 01 - acoustic feature representation
	feats_type = feats[1]  # RASTA-PLP 
	curr_lang = lang[0] # Tamil
	generate_acousticFeats(app_path, audio_path,feats_type,curr_lang) 
	
	#----------------- discovery ------------------- 
	# step 02 - discovery task 
	spoken_term_discovery(app_path, curr_lang,feats_type)

