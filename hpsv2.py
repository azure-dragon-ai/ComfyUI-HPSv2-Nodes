import torch
from PIL import Image
from PIL import ImageOps
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
class GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "get_size"

    CATEGORY = "Haojihui/Image"

    def get_size(self, image):
        _, height, width, _ = image.shape
        return (width, height)

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
    RETURN_TYPES = ("STRING", "FLOAT")

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
                torch.cuda.empty_cache()
            scores = hps_score[0]
        scores_str = str(scores)

        return (scores_str, scores)
    
class ImageScores:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "processor": ("PS_PROCESSOR",),
                "images": ("IMAGE",),
                "text_tokenizer": ("PS_TEXT_TOKENIZER",),
                "prompt": ("PS_PROMPT",),
                "orderby": (["asc", "desc"],),
                "device": (("cuda", "cpu"),),
            },
            "optional": {
                
            },
        }

    CATEGORY = "Haojihui/HPSv2"
    FUNCTION = "imageScores"
    RETURN_NAMES = ("SCORES", "SCORES1")
    RETURN_TYPES = ("STRING", "STRING")

    def imageScores(
        self,
        model,
        processor,
        images,
        text_tokenizer,
        prompt,
        orderby,
        device
    ):
        tokenizer = get_tokenizer('ViT-H-14')
        list_scores = list()
        for image in images:
            with torch.no_grad():
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    print(image)
                    print(text_tokenizer)
                    print(prompt)
                    numpy = image.numpy()
                    print(numpy.shape)
                    imageTensor = transforms.ToTensor()(numpy)
                    print("imageTensor ", imageTensor.shape)
                    image1 = transforms.ToPILImage()(imageTensor)

                    #image = Image.fromarray(numpy)
                    image_inputs = processor(image1).unsqueeze(0).to(device=device, non_blocking=True)

                    text_tokenizer = tokenizer([prompt]).to(device=device, non_blocking=True)
                    print(text_tokenizer)
                    outputs = model(image_inputs, text_tokenizer)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                scores = hps_score[0]
            scores_str = str(scores)
            list_scores.append(scores_str)
        torch.cuda.empty_cache()
        list_scores1 = []
        for i in range(len(list_scores)):
            list_scores1.append(list_scores[i])
        if orderby == "asc":
            list_scores1.sort()
        else:
            list_scores1.sort(reverse=True)

        return (list_scores, json.dumps(list_scores1))

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
                },
                "optional":
                {
                    "score": ("STRING", {"forceInput": True}),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Haojihui/Image"

    def save_images(self, images, filename_prefix="Hjh", score="", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        if score != "":
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
                    "score": ("STRING", {"forceInput": True}),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

    RETURN_TYPES = ()
    FUNCTION = "save_webp_images"

    OUTPUT_NODE = True

    CATEGORY = "Haojihui/Image"

    def save_webp_images(self, images, filename_prefix="Hjh", score="", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        if score != "":
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

class SaveWEBP:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required":
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "Hjh"}),
                     "lossless": ("BOOLEAN", {"default": True}),
                     "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                     "method": (list(s.methods.keys()),),
                     },
                "optional":
                {
                    "scores": ("STRING", {"forceInput": True}),
                    "orderby": (["asc", "desc"],),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Haojihui/Image"

    def save_images(self, images, filename_prefix, lossless, quality, method, scores=None, orderby=None,prompt=None, extra_pnginfo=None):
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
            if extra_pnginfo is not None:
                inital_exif = 0x010f
                for x in extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                    inital_exif -= 1

        c = len(pil_images)
        for i in range(0, c):
            score = ""
            if scores is not None:
                score = scores[i]
            if score != "":
                file = f"{filename}_{score}_{counter:05}_.webp"
            else:
                file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), exif=metadata, lossless=lossless, quality=quality, method=method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
                "score": score
            })
            counter += 1

        if orderby is not None:
            if orderby == "asc":
                results.sort(key=self.takeScore)
            else:
                results.sort(key=self.takeScore, reverse=True)
        animated = False
        return { "ui": { "images": results, "animated": (animated,) } }
    
    def takeScore(self, elem):
        return elem["score"]

class SaveAnimatedWEBP:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "fps": ("FLOAT", {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "lossless": ("BOOLEAN", {"default": True}),
                     "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                     "method": (list(s.methods.keys()),),
                     # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Haojihui/Image"

    def save_images(self, images, fps, filename_prefix, lossless, quality, method, num_frames=0, prompt=None, extra_pnginfo=None):
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
            if extra_pnginfo is not None:
                inital_exif = 0x010f
                for x in extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                    inital_exif -= 1

        if num_frames == 0:
            num_frames = len(pil_images)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], exif=metadata, lossless=lossless, quality=quality, method=method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        animated = num_frames != 1
        return { "ui": { "images": results, "animated": (animated,) } }

#ComfyUI原始SaveImage节点变更存图格式
class SaveImageWebp:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "lossless": ("BOOLEAN", {"default": False}),
                     "quality": ("INT", {"default": 85, "min": 0, "max": 100})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Haojihui/Image"

    def save_images(self, images, lossless, quality, filename_prefix, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = img.getexif()
            
            if not args.disable_metadata:
                if prompt is not None:
                    metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
                if extra_pnginfo is not None:
                    inital_exif = 0x010f
                    for x in extra_pnginfo:
                        metadata[inital_exif] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                        inital_exif -= 1

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.webp"
            img.save(os.path.join(full_output_folder, file),method=6, exif=metadata, lossless=lossless, quality=quality, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

#指定短边保持比例缩放图片
def img_to_tensor(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor

def tensor_to_img(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
    return img

class ScaleShort:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 16}),
                "crop_face": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale_short"

    CATEGORY = "Haojihui/Image"

    def scale_short(self, image, size, crop_face):
        input_image = tensor_to_img(image)
        short_side = min(input_image.width, input_image.height)
        resize = float(short_side / size)
        new_size = (int(input_image.width // resize), int(input_image.height // resize))
        input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
        if crop_face:
            new_width = int(np.shape(input_image)[1] // 32 * 32)
            new_height = int(np.shape(input_image)[0] // 32 * 32)
            input_image = input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)
        return (img_to_tensor(input_image),)

NODE_CLASS_MAPPINGS = {
    "ScaleShort": ScaleShort,
    "SaveImageWebp": SaveImageWebp,
    "GetImageSize": GetImageSize,
    "HaojihuiHPSv2Loader": Loader,
    "HaojihuiHPSv2ImageProcessor": ImageProcessor,
    "HaojihuiHPSv2TextProcessor": TextProcessor,
    "HaojihuiHPSv2ImageScore": ImageScore,
    "HaojihuiHPSv2ImageScores": ImageScores,
    "HaojihuiHPSv2SaveImage": SaveImage,
    "HaojihuiHPSv2SaveWebpImage": SaveWebpImage,
    "HaojihuiHPSv2SaveWEBP": SaveWEBP,
    "HaojihuiHPSv2SaveAnimatedWEBP": SaveAnimatedWEBP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HaojihuiHPSv2Loader": "Loader",
    "HaojihuiHPSv2ImageProcessor": "Image Processor",
    "HaojihuiHPSv2TextProcessor": "Text Processor",
    "HaojihuiHPSv2ImageScore": "ImageScore",
    "HaojihuiHPSv2ImageScores": "ImageScores",
    "HaojihuiHPSv2SaveImage": "SaveImage",
    "HaojihuiHPSv2SaveWebpImage": "SaveWebpImage",
    "HaojihuiHPSv2SaveWEBP": "SaveWEBP",
    "HaojihuiHPSv2SaveAnimatedWEBP": "SaveAnimatedWEBP",
}
