import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .vision_transformer import build_vision_tower
from .text_transformer import build_text_tower
from .conch_v1_5 import build_conch
from .configuration_titan import TitanConfig

class Titan(PreTrainedModel):
    config_class = TitanConfig

    def __init__(self, config: TitanConfig, *model_args, **model_kwargs):
        super().__init__(config)

        self.vision_encoder = build_vision_tower(config.vision_config)
        self.text_encoder = build_text_tower(config.text_config)
        self.conch_config = config.conch_config

    def return_conch(self):
        model, eval_transform = build_conch(self.conch_config)
        return model, eval_transform

    def encode_slide_from_patch_features(self, patch_features: torch.Tensor, patch_coords: torch.Tensor, patch_size_lv0: int) -> torch.Tensor:
        '''
        encode whole-slide image using patch features
        Args:
            patch_features: torch.Tensor, shape (1, N, C)
            patch_coords: torch.Tensor, shape (1, N, 2)
            patch_size_lv0: int, patch size at level 0 (1024 if slide is 40x, 512 if slide is 20x)
        '''
        slide_embedding = self.vision_encoder(patch_features, patch_coords, patch_size_lv0, no_proj=True)
        return slide_embedding
    
    def encode_text(self, input_ids: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        '''
        Args:
            input_ids: torch.Tensor, shape (B, L)
        '''
        input_ids = input_ids[:, :-1] # make space for CLS token
        text_latent, _ = self.text_encoder(input_ids)
        if normalize:
            text_latent = F.normalize(text_latent, dim=-1)
        return text_latent

    def zero_shot_classifier(self, classnames, templates, device=None):
        """
        classnames: list of lists of classnames (one list of classnames per class)
        templates: list of templates 
        """
        zeroshot_weights = []
        for classnames_for_class in classnames:
            embeddings_for_class = []
            for classname in classnames_for_class:
                texts = [template.replace('CLASSNAME', classname) for template in templates]
                tokenized_prompts = self.text_encoder.tokenizer(texts)  # Tokenize with custom tokenizer
                tokenized_prompts = tokenized_prompts.to(device)
                classname_embeddings = self.encode_text(tokenized_prompts, normalize=True)
                embeddings_for_class.append(classname_embeddings)

            class_embedding = torch.stack(embeddings_for_class, dim=0)
            # over all templates and classnames
            class_embedding = class_embedding.mean(dim=(0, 1))
            class_embedding /= class_embedding.norm()

            # class_embedding: [embedding_dim]
            zeroshot_weights.append(class_embedding)

        # zeroshot_weights: [embedding_dim, num_classes]
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights
    
    def zero_shot(self, slide_embedding: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
        '''
        zero-shot inference for whole-slide image
        Args:
            slide_embedding: torch.Tensor, shape (1, D)
            classifier: torch.Tensor, shape (D, num_classes)
        '''
        device = slide_embedding.device

        slide_embedding = slide_embedding.float().to(device).unsqueeze(0)
        slide_embedding = slide_embedding @ self.vision_encoder.proj
        slide_embedding = F.normalize(slide_embedding, dim=-1)

        logits = slide_embedding @ classifier
        probs = F.softmax(logits, dim=-1)
        return probs
