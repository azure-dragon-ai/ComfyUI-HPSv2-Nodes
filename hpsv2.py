import torch
from PIL import Image
import numpy as np
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import os
from torchvision import transforms
import folder_paths
from comfy.cli_args import args
from PIL.PngImagePlugin import PngInfo
import json
import webp

# set HF_ENDPOINT=https://hf-mirror.com
class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "HPSv2\HPSv2Models\HPS_v2_compressed.pt"}),
                "device": (("cuda", "cpu"),),
                "dtype": (("float16", "bfloat16", "float32"),),
            },
        }

    CATEGORY = "Haojihui/HPSv2"
    FUNCTION = "load"
    RETURN_NAMES = ("MODEL", "TOKENIZER", "PROCESSOR")
    RETURN_TYPES = ("PS_MODEL", "PS_TOKENIZER", "PS_PROCESSOR")

    def load(self, path, device, dtype):
        # os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
        # os.environ['HPS_ROOT'] = "HPSv2\HPSv2Models"
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
        print(images.shape)
        numpy = images[0].numpy()
        print(numpy.shape)
        imageTensor = transforms.ToTensor()(numpy)
        print("imageTensor ", imageTensor.shape)
        image = transforms.ToPILImage()(imageTensor)

        
        #image = Image.fromarray(numpy)


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
    RETURN_NAMES = ("TEXT_TOKENIZER", "PROMPT")
    RETURN_TYPES = ("PS_TEXT_TOKENIZER", "PS_PROMPT")

    def process(self, tokenizer, device, text):
        prompt = text
        print(prompt)
        ret = tokenizer([prompt]).to(device=device, non_blocking=True)
        print(ret)
        return (
            ret, prompt
        )


class ImageScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "image_inputs": ("IMAGE_INPUTS",),
                "text_tokenizer": ("PS_TEXT_TOKENIZER",),
                "prompt": ("PS_PROMPT",),
                "device": (("cuda", "cpu"),),
            },
            "optional": {
                
            },
        }

    CATEGORY = "Haojihui/HPSv2"
    FUNCTION = "imageScore"
    RETURN_NAMES = ("SCORES", "SCORES1")
    RETURN_TYPES = ("PS_SCORES", "STRING")

    def imageScore(
        self,
        model,
        image_inputs,
        text_tokenizer,
        prompt,
        device
    ):
        tokenizer = get_tokenizer('ViT-H-14')
        with torch.no_grad():
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                print(image_inputs)
                print(text_tokenizer)
                print(prompt)
                text_tokenizer = tokenizer([prompt]).to(device=device, non_blocking=True)
                print(text_tokenizer)
                outputs = model(image_inputs, text_tokenizer)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            scores = hps_score[0]
        scores_str = str(scores)

        return (scores_str, scores_str)

class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": 
                {
                    "images": ("IMAGE", ),
                    "filename_prefix": ("STRING", {"default": "Hjh"}),
                    "score1": ("STRING", {"forceInput": True}),
                },
                "optional":
                {
                    "score": ("PS_SCORES",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Haojihui/HPSv2"

    def save_images(self, images, filename_prefix="Hjh", score="", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        filename_prefix += "_" + score
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                metadata.add_text("score", json.dumps(score))
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
    
class SaveWebpImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": 
                {
                    "images": ("IMAGE", ),
                    "filename_prefix": ("STRING", {"default": "Hjh"}),
                },
                "optional":
                {
                    "score": ("PS_SCORES",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

    RETURN_TYPES = ()
    FUNCTION = "save_webp_images"

    OUTPUT_NODE = True

    CATEGORY = "Haojihui/HPSv2"

    def save_webp_images(self, images, filename_prefix="Hjh", score="", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        filename_prefix += "_" + score
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.webp"
            webp.save_image(img, os.path.join(full_output_folder, file), quality=80)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
    "HaojihuiHPSv2Loader": Loader,
    "HaojihuiHPSv2ImageProcessor": ImageProcessor,
    "HaojihuiHPSv2TextProcessor": TextProcessor,
    "HaojihuiHPSv2ImageScore": ImageScore,
    "HaojihuiHPSv2SaveImage": SaveImage,
    "HaojihuiHPSv2SaveWebpImage": SaveWebpImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HaojihuiHPSv2Loader": "Loader",
    "HaojihuiHPSv2ImageProcessor": "Image Processor",
    "HaojihuiHPSv2TextProcessor": "Text Processor",
    "HaojihuiHPSv2ImageScore": "ImageScore",
    "HaojihuiHPSv2SaveImage": "SaveImage",
    "HaojihuiHPSv2SaveWebpImage": "SaveWebpImage",
}
