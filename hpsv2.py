import torch
from comfy.model_management import InterruptProcessingException
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

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
        return (
            processor(images).unsqueeze(0).to(device=device, non_blocking=True),
        )


class TextProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokenizer": ("TOKENIZER",),
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


class Selector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "image_inputs": ("IMAGE_INPUTS",),
                "text_inputs": ("TEXT_INPUTS",),
                "threshold": ("FLOAT", {"max": 1, "step": 0.001}),
                "limit": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
                "masks": ("MASK",),
            },
        }

    CATEGORY = "Haojihui/HPSv2"
    FUNCTION = "select"
    RETURN_NAMES = ("SCORES", "IMAGES", "LATENTS", "MASKS")
    RETURN_TYPES = ("STRING", "IMAGE", "LATENT", "MASK")

    def select(
        self,
        model,
        image_inputs,
        text_inputs,
        threshold,
        limit,
        images=None,
        latents=None,
        masks=None,
    ):
        with torch.inference_mode():
            image_inputs = image_inputs.to(model.device, dtype=model.dtype)
            image_embeds = model.get_image_features(image_inputs)
            image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)

            text_inputs = text_inputs.to(model.device)
            text_embeds = model.get_text_features(text_inputs)
            text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)

            scores = (text_embeds.float() @ image_embeds.float().T)[0]

            if scores.shape[0] > 1:
                scores = model.logit_scale.exp() * scores
                scores = torch.softmax(scores, dim=-1)

        scores = scores.cpu().tolist()
        scores = {k: v for k, v in enumerate(scores) if v >= threshold}
        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)[:limit]
        scores_str = ", ".join([str(round(v, 3)) for k, v in scores])

        if images is not None:
            images = [images[v[0]] for v in scores]
            images = torch.stack(images) if images else None

        if latents is not None:
            latents = latents["samples"]
            latents = [latents[v[0]] for v in scores]
            latents = {"samples": torch.stack(latents)} if latents else None

        if masks is not None:
            masks = [masks[v[0]] for v in scores]
            masks = torch.stack(masks) if masks else None

        if images is None and latents is None and masks is None:
            raise InterruptProcessingException()

        return (scores_str, images, latents, masks)


NODE_CLASS_MAPPINGS = {
    "HaojihuiHPSv2Loader": Loader,
    "HaojihuiHPSv2ImageProcessor": ImageProcessor,
    "HaojihuiHPSv2TextProcessor": TextProcessor,
    "HaojihuiHPSv2Selector": Selector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HaojihuiHPSv2Loader": "Loader",
    "HaojihuiHPSv2ImageProcessor": "Image Processor",
    "HaojihuiHPSv2TextProcessor": "Text Processor",
    "HaojihuiHPSv2Selector": "Selector",
}
