import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np
import cv2

from utils.file_utils import save_hdf5
from dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.foundation_model.builder1 import get_encoder

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

def find_k_for_cumulative_variance(S, threshold=0.90):
	singular_values = S.flatten()
	cumulative_variance = torch.cumsum(singular_values ** 2, dim=0) / torch.sum(singular_values ** 2)
	k = (cumulative_variance >= threshold).nonzero(as_tuple=True)[0].min()
	return k

def compute_w_loader(output_path, loader, model, verbose = 0, is_svd = False):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			# savePath = r'/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/print/'
			# rgb_image = batch[count, :, :, :].permute(1, 2, 0)
			# image = 255 * (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
			# save_image = torch.cat((image[:,:,2].unsqueeze(2),image[:,:,1].unsqueeze(2),image[:,:,0].unsqueeze(2)),dim=2).cpu().numpy()
			# cv2.imwrite(savePath + 'Image_' + str(count) + '_' + 'src_image_RGB.jpg', save_image)

			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			# svd
			if is_svd:
				k_d = 512
				if batch.shape[0] < k_d:
					quo = k_d // batch.shape[0]
					rem = k_d % batch.shape[0]
					batch_ex = batch
					for i in range(quo-1):
						batch_ex = torch.cat((batch_ex, batch), dim=0)
					for j in range(rem):
						batch_ex = torch.cat((batch_ex, batch[j, :].unsqueeze(0)), dim=0)
				curU, curS, curV = torch.svd(batch_ex.reshape(batch_ex.shape[0], -1))
				features_add = torch.matmul(torch.matmul(curU[:, :k_d], curS[:k_d].unsqueeze(1).expand_as(curU[:k_d, :].permute(1, 0))), curV[:k_d, :].permute(1, 0))[:batch.shape[0], :]
				features_add = features_add.cpu().numpy().astype(np.float32)

				asset_dict = {'features': features, 'features_add': features_add, 'coords': coords}
			else:
				asset_dict = {'features': features, 'coords': coords}

			save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
			mode = 'a'

	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
# parser.add_argument('--data_h5_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/svs_patch/')
# parser.add_argument('--data_slide_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/svs/')
# parser.add_argument('--csv_path', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/LGG-GBM.csv')
# parser.add_argument('--feat_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/patch_feature')
parser.add_argument('--data_h5_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch/')
parser.add_argument('--data_slide_dir', type=str, default='/data1/xiamy/PCa-EPE/WSI/')
parser.add_argument('--csv_path', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/PCa-EPE_use.csv')
parser.add_argument('--feat_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_prov-gigapath/')
parser.add_argument('--is_svd', type=str, default= False)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--model_name', type=str, default='prov-gigapath', choices=['resnet50_trunc', 'uni_v1', 'H-optimus-0', 'prov-gigapath', 'uni2-h', 'conch_v1', 'conch_v1_5', 'CHIEF-Ctranspath', 'MUSK'])
# resnet50-1024
# uni-1024
# uni2-1536
# conch-512
# titan
# CHIEF-768
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

		from pathlib import Path

		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		if Path(slide_file_path).exists() is False:
			slide_file_path = os.path.join(args.data_slide_dir, slide_id+'.ndpi')
			if Path(slide_file_path).exists() is False:
				slide_file_path = os.path.join(args.data_slide_dir, slide_id+'.tiff')

		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1, is_svd = args.is_svd)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			if args.is_svd:
				features_add = file['features_add'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
		if args.is_svd:
			features_add = torch.from_numpy(features_add)
			torch.save(features_add, os.path.join(args.feat_dir, 'pt_files', bag_base + '_add.pt'))



