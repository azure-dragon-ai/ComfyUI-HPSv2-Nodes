import torch
from PIL import Image
import numpy as np
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import os
from torchvision import transforms

# set HF_ENDPOINT=https://hf-mirror.com
class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "c:\Work\AI-Service\Score\HPSv2\HPSv2Models\HPS_v2_compressed.pt"}),
                "device": (("cuda", "cpu"),),
                "dtype": (("float16", "bfloat16", "float32"),),
            },
        }

    CATEGORY = "Haojihui/HPSv2"
    FUNCTION = "load"
    RETURN_NAMES = ("MODEL", "TOKENIZER", "PROCESSOR")
    RETURN_TYPES = ("PS_MODEL", "PS_TOKENIZER", "PS_PROCESSOR")

    def load(self, path, device, dtype):
        os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
        os.environ['HPS_ROOT'] = "c:\Work\AI-Service\Score\HPSv2\HPSv2Models"
        dtype = torch.float32 if device == "cpu" else getattr(torch, dtype)
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        #model = AutoModel.from_pretrained(path, torch_dtype=dtype).eval().to(device)
        #processor = AutoProcessor.from_pretrained(path)

        # HPS_v2_compressed.pt
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = get_tokenizer('ViT-H-14')
        model = model.to(device)
        model.eval()
        return (model, tokenizer, preprocess_val)


class ImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processor": ("PS_PROCESSOR",),
                "device": (("cuda", "cpu"),),
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "Haojihui/HPSv2"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE_INPUTS",)

    def process(self, processor, device, images):
        image = transforms.ToPILImage()(images)

        return (
            processor(image).unsqueeze(0).to(device=device, non_blocking=True),
        )


class TextProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokenizer": ("PS_TOKENIZER",),
                "device": (("cuda", "cpu"),),
                "text": ("STRING", {"multiline": True}),
            },
        }

    CATEGORY = "Haojihui/HPSv2"
    FUNCTION = "process"
    RETURN_TYPES = ("TEXT_INPUTS",)

    def process(self, tokenizer, device, text):
        return (
            tokenizer([text]).to(device=device, non_blocking=True)
        )


class ImageScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "image_inputs": ("IMAGE_INPUTS",),
                "text_inputs": ("TEXT_INPUTS",),
            },
            "optional": {
                
            },
        }

    CATEGORY = "Haojihui/HPSv2"
    FUNCTION = "imageScore"
    RETURN_NAMES = ("SCORES", "SCORES1")
    RETURN_TYPES = ("STRING", "STRING")

    def imageScore(
        self,
        model,
        image_inputs,
        text_inputs,
    ):
        with torch.no_grad():
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image_inputs, text_inputs)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()

        scores = "score:" . hps_score[0]

        return (scores, scores)


NODE_CLASS_MAPPINGS = {
    "HaojihuiHPSv2Loader": Loader,
    "HaojihuiHPSv2ImageProcessor": ImageProcessor,
    "HaojihuiHPSv2TextProcessor": TextProcessor,
    "HaojihuiHPSv2ImageScore": ImageScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HaojihuiHPSv2Loader": "Loader",
    "HaojihuiHPSv2ImageProcessor": "Image Processor",
    "HaojihuiHPSv2TextProcessor": "Text Processor",
    "HaojihuiHPSv2ImageScore": "ImageScore",
}
