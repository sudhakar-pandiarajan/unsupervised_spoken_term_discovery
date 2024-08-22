'''
	Author: Sudhakar. P 
	Updated on : April 30, 2024
'''
import pickle
from tqdm import tqdm
import sys
import os
from os import path
import glob
import numpy as np
from scipy.spatial import distance
class Discovery: 
	def cosine_sim(self,a,b): 
		'''
			Compute cosine similarity 
		'''
		dist = distance.cosine(a,b)
		return 1-dist
	
	def load_acoustic_feats_dict(self,feats_path):
		feats_dict = {}
		for fname in tqdm(sorted(glob.glob(feats_path))):
			file_name= path.splitext(path.split(fname)[-1])[0]
			feats = np.load(fname)
			feats_dict[file_name] = feats
		return feats_dict
	
	def diag_sim(self,mat, sim_thresh=9): 
		'''
			captures the potential pattern match propagations in the diagonal
		'''
		C = np.zeros((mat.shape[0],  mat.shape[1]), int)
		#print(C.shape)
		match_score=1
		for i in range(0, mat.shape[0]): 
			for j in range(0, mat.shape[1]):
				if(mat[i,j]==1):
					if(i>0 and j>0):
						match =  match_score+C[i - 1, j - 1]
						C[i, j] = match
					elif(i==0 or j==0):
						C[i,j] = match_score

		row = mat.shape[0]
		clm = mat.shape[1]
		r_index = (row-1)*-1
		c_index = clm-1
		diags = (r_index*-1)+1+(c_index)
		h_cost = np.zeros((diags))
		h_cost_l =[]
		for i in range(r_index,0,1): 
			diags = C.diagonal(i)
			grad = np.array(diags)
			seq_count = np.sum(grad)
			if(seq_count>=sim_thresh):
				h_cost_l.append((i,seq_count)) # add the diagonal and cost

		h_cost_r=[]
		for i in range(0,c_index,1): 
			diags = C.diagonal(i)
			grad = np.array(diags)
			seq_count = np.sum(grad)
			if(seq_count>=sim_thresh):
				h_cost_r.append((i, seq_count))
			
		return h_cost_l, h_cost_r, C
	
	def compute_word_pair_match(self, feats_path, result_path, sim_thre=0.97, depth_thre=9):
		'''
			Compute word pair match in-memory
		'''
		feats_dict = self.load_acoustic_feats_dict(feats_path)
		f_names = [key for key in feats_dict.keys()]
		f_count = len(f_names)
		print("Total files loaded : ", f_count)
		doc_pairs = [(f_names[i],f_names[j]) for i in range(0,f_count,1) for j in range(1,f_count,1) if j >i]
		print("Total documents compared : ", len(doc_pairs))
		#for dx, dy in tqdm(doc_pairs): 
		for Dx, Dy in tqdm(doc_pairs): # make combinations
			print("Computing similarity between {} and {}...".format(Dx, Dy))
			dx_name="{}".format(Dx)
			dy_name="{}".format(Dy)
			feats_spkr1 = feats_dict[dx_name].T
			feats_spkr2 = feats_dict[dy_name].T
			
			c, r1 = feats_spkr1.shape # dx
			c, r2 = feats_spkr2.shape # dy
			
			sim_mat = np.zeros((r1, r2))
			#norm_feats1 = normalize(feats_spkr1, axis=0)
			#norm_feats2 = normalize(feats_spkr2, axis=0)
			norm_feats1 = feats_spkr1
			norm_feats2 = feats_spkr2
			for i in range(r1): 
				for j in range(r2): 
					x = norm_feats1[:,i]
					y = norm_feats2[:,j]
					sim_mat[i,j] = self.cosine_sim(x,y) # cosine similarity
			r, c = sim_mat.shape
			sim_out = np.where(sim_mat>sim_thre,1,0) # for cosine sim
			
			h_cost_l, h_cost_r, C = self.diag_sim(sim_out, depth_thre)
			word_dia_l = h_cost_l
			word_dia_r = h_cost_r
			wordpair_match=[]
			x = 0 # meant for document X index (in rowwise in the matrix)
			for di in range(len(word_dia_l)):
				d, c = word_dia_l[di]
				dia_ele = C.diagonal(d) # for lower diagonal starts from -1
				dx=d*-1
				max_limit = len(dia_ele)
				i=0
				while(i<max_limit): 
					if(dia_ele[i]>0): 
						st_x = x+i 
						st_y= i
						depth = 1
						while(dia_ele[i]>0 and i<len(dia_ele)-1):
							depth =depth+1
							i=i+1
						end_x= st_x+depth-1
						end_y =st_y+depth-1
						if(depth>=depth_thre):
							wordpair_match.append((st_x,end_x,st_y,end_y)) # (x-st,x-end, y-st, y-end)
					i=i+1
				
			y=0 # meant for document Y index (in columnwise)
			st_x = 0
			st_y = 0
			end_x =0
			end_y = 0
			for dj in  range(len(word_dia_r)): 
				dia, co = word_dia_r[dj]
				dia_ele = C.diagonal(dia) # for main and upper diagonal starts from 0 
				#print(di, co)
				y=dia
				max_limit=len(dia_ele)
				j = 0
				while(j<max_limit):
					if(dia_ele[j]>0): 
						st_x= j
						st_y= y+j
						depth=1
						while(dia_ele[j]>0 and j<len(dia_ele)-1):
							depth =depth+1
							j=j+1
						end_x=st_x+depth-1
						end_y=st_y+depth-1
						if(depth>=depth_thre):
							wordpair_match.append((st_x,end_x,st_y, end_y))
					j=j+1
				

			#print(wordpair_match)
			found = len(wordpair_match)
			if(found>0):
				print("{} pairs found between {} and {}.".format(found, Dx, Dy))
				Dx = Dx.replace("-","_")
				Dy = Dy.replace("-","_")
				result_file = path.join(result_path,"{}-{}.pickle".format(Dx,Dy))
				#word_pair_clus[(dx,dy)] = wordpair_match
				with open(result_file,'wb') as fout: 
					pickle.dump(wordpair_match,fout)
