import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class SMSynthDataset(Dataset):
	def __init__(self, filename, instruments=[], num_frames=0, pitches=[]):
		data = pickle.load(open(filename, 'rb'))
		names = list(data.keys())
		self.names = []
		self.labels = []
		self.pitches = []
		self.velocities = []
		self.ccs = []

		for it,k in enumerate(names):
			temp_file = data[k]
			p = temp_file['pitch']
			v = temp_file['velocity']
			cc_curr = temp_file['cc']
			
			if (p in pitches):
				fc = 0
				if(cc_curr.shape[0] > num_frames):
					num_iters = num_frames
				else:
					num_iters = cc_curr.shape[0]
					
				for f in range(num_iters):
					self.names.append(k + '_frame_'+str(fc + 1))
					fc = fc + 1
					self.labels.append(k.split('_')[0])
					self.pitches.append(int(p))
					self.velocities.append(int(v))
					self.ccs.append(cc_curr[f,:])


        # if data[names[0]]['cc'].ndim == 1:
        #     max_num_frames = 1
        # else:
        #     max_num_frames = data[names[0]]['cc'].shape[0]
        # if num_frames==0 or num_frames > max_num_frames:
        #     num_frames = max_num_frames
        # for name in names:
        #     label = name.split('_')[0]
        #     if (len(instruments)==0) or (label in instruments):
        #         if (int(data[name]['pitch']) in pitches) or (len(pitches)==0):
        #             try:
        #                 if data[name]['cc'].ndim == 1:
        #                     temp = np.expand_dims(data[name]['cc'],axis = 0)
        #             except KeyError:
        #                 print("KeyError")
        #                 continue
        #             else:
        #                 temp = data[name]['cc']
        #             for i in range(num_frames):
        #                 if np.count_nonzero(temp[i,1:])==0:
        #                     continue
        #                 self.names.append(name+'-frame'+str(i))
        #                 self.labels.append(label)
        #                 self.pitches.append(int(data[name]['pitch']))
        #                 self.velocities.append(int(data[name]['velocity']))
        #                 if max_num_frames==1:
        #                     self.ccs.append(temp[0,:])
        #                 else:
        #                     self.ccs.append(temp[i,:])


	def __len__(self):
		return len(self.names)

	def __getitem__(self, idx):
		return self.labels[idx], torch.tensor(self.pitches[idx]), torch.tensor(self.velocities[idx]), torch.tensor(self.ccs[idx])
