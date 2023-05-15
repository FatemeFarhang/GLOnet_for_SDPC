import numpy as np
import pandas as pd
import torch

class MatDatabase(object):
	"""docstring for MatDatabase
		Parameters: 
			material_key: list of material names
	"""
	def __init__(self, material_key):
		super(MatDatabase, self).__init__()
		self.material_key = material_key
		self.num_materials = len(material_key)
		self.mat_database = self.build_database()

	def build_database(self):
		mat_database = {}
		
		#%% Read in the dispersion data of each material
		for i in range(self.num_materials):
			file_name = './material_database/mat_' + self.material_key[i] + '.xlsx'
			
			try: 
				A = np.array(pd.read_excel(file_name))
				mat_database[self.material_key[i]] = (A[:, 0], A[:, 1], A[:, 2])
			except NameError:
				print('The material database does not contain', self.material_key[i])
		return mat_database


	# def interp_wv(self, wv_in, material_key, ignoreloss = False):
	def interp_wv(self, wv_in):
		'''
			parameters
				wv_in (tensor) : number of wavelengths
				material_key (list) : number of materials

			return
				refractive indices (tensor or tuple of tensor) : number of materials x number of wavelengths
		'''

		n_data_L = np.zeros((1,wv_in.size(0)))
		n_data_H = np.zeros((1,wv_in.size(0)))

		n_data_D = np.zeros((1, wv_in.size(0)))
		mat_sio = self.mat_database['SiO2']
		mat_sin = self.mat_database['SiN']
		n_data_L = np.interp(wv_in, mat_sio[0], mat_sio[1])
		n_data_H = np.interp(wv_in, mat_sin[0], mat_sin[1])

		n_data_D = np.interp(wv_in, mat_sio[0], mat_sio[1])
		n_data = np.vstack((n_data_L,n_data_H, n_data_D))
        
		return torch.tensor(n_data).cuda() if torch.cuda.is_available() else torch.tensor(n_data)






		