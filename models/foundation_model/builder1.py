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
    elif model_name == 'H-optimus-0':
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
            'reg_tokens': 4,
            'dynamic_img_size': True
        }
        model = timm.create_model(**timm_kwargs)
        checkpoint_path = "/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/H-optimus-0_model/pytorch_model.bin"
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        xxx=0
    elif model_name == 'prov-gigapath':
        from os import makedirs
        from os.path import join
        from huggingface_hub import hf_hub_download
        import json
        from models.foundation_model.gigapath.slide_encoder import create_model as create_slide_encoder
        os.environ["HF_TOKEN"] = "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"
        assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"
        local_dir = "/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/prov-gigapath_model/"
        from os import makedirs
        makedirs(local_dir, exist_ok=True)
        # hf_hub_download(
        #     "prov-gigapath/prov-gigapath",
        #     filename="pytorch_model.bin",
        #     local_dir=local_dir,
        #     force_download=True
        # )
        # hf_hub_download(
        #     "prov-gigapath/prov-gigapath",
        #     filename="slide_encoder.pth",
        #     local_dir=local_dir,
        #     force_download=True
        # )
        # hf_hub_download(
        #     "prov-gigapath/prov-gigapath",
        #     filename="config.json",
        #     local_dir=local_dir,
        #     force_download=True
        # )
        config = json.load(open(join(local_dir, "config.json")))
        model = timm.create_model(
            model_name=config['architecture'],
            checkpoint_path=join(local_dir, "pytorch_model.bin"),
            **config["model_args"]
        )
        # slide_encoder = create_slide_encoder(
        #     pretrained=join(local_dir, "slide_encoder.pth"),
        #     model_arch="gigapath_slide_enc12l768d",
        #     in_chans=1536,
        #     global_pool=True
        # )
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
    elif model_name == 'MUSK':
        from models.foundation_model.musk import utils
        model = timm.models.create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')

    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms