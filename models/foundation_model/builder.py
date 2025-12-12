import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms


def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        UNI_CKPT_PATH = '/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/UNI_model/pytorch_model.bin'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'uni2-h':
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        model = timm.create_model(**timm_kwargs)
        model.load_state_dict(torch.load(os.path.join(
            '/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/UNI2-h_model/',
            "pytorch_model.bin"), map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        # use timm 0.9.8 pip install timm==0.9.8
        CONCH_CKPT_PATH = '/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/CONCH_model/pytorch_model.bin'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        # timm==1.0.3
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        # titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', local_files_only=True, trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    elif model_name == 'CHIEF-Ctranspath':
        # use pip install timm-0.5.4.tar
        from dataProcess.ctran import ctranspath
        model = ctranspath()
        model.head = torch.nn.Identity()
        td = torch.load(r'/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/CHIEF_model/CHIEF_CTransPath.pth')
        model.load_state_dict(td['model'], strict=True)

        # from models.CHIEF import CHIEF
        # model = CHIEF(size_arg="small", dropout=True, n_classes=2)
        # td = torch.load(r'/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/CHIEF_model/CHIEF_pretraining.pth')
        # model.load_state_dict(td, strict=True)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms