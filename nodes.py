import llama_cpp
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import folder_paths
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
from collections.abc import Callable
import comfy.model_management as mm
import os
import json


# From (https://github.com/gokayfem/ComfyUI_VLM_nodes/blob/1ca496c1c8e8ada94d7d2644b8a7d4b3dc9729b3/nodes/qwen2vl.py)
# Apache 2.0 License


# Read a JSON file and convert to a Python dictionary.
def read_joycaption_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"[wo_joycaption_comfyui] Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"[wo_joycaption_comfyui] Error: Invalid JSON format in '{file_path}'.")
        return None

joycaption_node_path = os.path.dirname(os.path.realpath(__file__))
joycaption_config = read_joycaption_config(os.path.join(joycaption_node_path, "joycaption_config.json"))

# Store the actual data structures from the config
CAPTION_TYPE_CONFIG_MAP = joycaption_config["CAPTION_TYPE_MAP"] # dictionary
CAPTION_LENGTH_CHOICES_LIST = list(joycaption_config["CAPTION_LENGTH_CHOICES"]) # list of strings
MEMORY_EFFICIENT_CONFIGS_DICT = joycaption_config["MEMORY_EFFICIENT_CONFIGS"] # dictionary
OPTIONS_CONFIG_MAP = joycaption_config["EXTRA_MAP"] # dictionary

# Derive lists of keys/choices specifically for UI elements (dropdowns)
CAPTION_TYPE_CHOICES_KEYS = list(CAPTION_TYPE_CONFIG_MAP.keys())
# CAPTION_LENGTH_CHOICES_LIST is already suitable for UI choices
MEMORY_EFFICIENT_MODES_KEYS = list(MEMORY_EFFICIENT_CONFIGS_DICT.keys())


def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
    prompt_templates_list = []

    if caption_type not in CAPTION_TYPE_CONFIG_MAP:
        print(f"[wo_joycaption_comfyui] Warning: Unknown caption_type '{caption_type}'. Attempting to use default.")
        if not CAPTION_TYPE_CONFIG_MAP or not list(CAPTION_TYPE_CONFIG_MAP.keys()):
            print(f"[wo_joycaption_comfyui] Error: CAPTION_TYPE_CONFIG_MAP is empty or invalid. Cannot determine default prompt.")
            return "Error: CAPTION_TYPE_CONFIG_MAP is misconfigured."
        
        default_template_key = list(CAPTION_TYPE_CONFIG_MAP.keys())[0]
        print(f"[wo_joycaption_comfyui] Warning: Using default caption type '{default_template_key}'.")
        prompt_templates_list = CAPTION_TYPE_CONFIG_MAP.get(default_template_key, [])
        
        if not prompt_templates_list:
             print(f"[wo_joycaption_comfyui] Error: Default caption type '{default_template_key}' has no templates.")
             return f"Error: No templates for default type {default_template_key}."
    else:
        prompt_templates_list = CAPTION_TYPE_CONFIG_MAP.get(caption_type, [])
        if not prompt_templates_list:
            print(f"[wo_joycaption_comfyui] Error: Caption type '{caption_type}' has no templates defined.")
            return f"Error: No templates for caption type '{caption_type}'."

    # Determine which template to use from the list based on caption_length
    # Template indices: 0 for "any", 1 for "{word_count}", 2 for "{length}"
    chosen_template_idx = 0 
    actual_caption_length_str = str(caption_length) 

    if actual_caption_length_str.isdigit():
        if len(prompt_templates_list) > 1:
            chosen_template_idx = 1  # Use template with {word_count}
        else:
            print(f"[wo_joycaption_comfyui] Warning: Not enough templates for '{caption_type}' to use specific word count. Using general template (index 0).")
            chosen_template_idx = 0
    elif actual_caption_length_str != "any":  # Descriptive length like "short", "long"
        if len(prompt_templates_list) > 2:
            chosen_template_idx = 2  # Use template with {length}
        else:
            print(f"[wo_joycaption_comfyui] Warning: Not enough templates for '{caption_type}' to use descriptive length. Using general template (index 0).")
            chosen_template_idx = 0
    
    if chosen_template_idx >= len(prompt_templates_list):
        print(f"[wo_joycaption_comfyui] Warning: Template index {chosen_template_idx} out of bounds for '{caption_type}' (list size {len(prompt_templates_list)}). Falling back to index 0.")
        chosen_template_idx = 0
        if not prompt_templates_list : 
            print(f"[wo_joycaption_comfyui] Error: Critical - no templates available for {caption_type} after fallback.")
            return f"Error: Critical - no templates available for {caption_type}."
    
    selected_prompt_template_str = prompt_templates_list[chosen_template_idx]
    
    formatted_base_prompt = selected_prompt_template_str # Initialize with the chosen template string
    name_to_insert = name_input or "{NAME}" # Use placeholder if name_input is empty

    try:
        if chosen_template_idx == 1: # Template expects {word_count} and possibly {name}
            # Example: "Write a detailed description for this image in {word_count} words or less."
            # Ensure all expected keys by this specific template are provided.
            # If template doesn't have {name}, .format will ignore extra 'name' argument.
            # If template *requires* {name} but it's missing from string, it's a template design issue.
            # For safety, check if placeholders exist before formatting or use individual replace.
            temp_prompt = selected_prompt_template_str
            if "{name}" in temp_prompt:
                temp_prompt = temp_prompt.replace("{name}", name_to_insert)
            if "{word_count}" in temp_prompt:
                temp_prompt = temp_prompt.replace("{word_count}", actual_caption_length_str)
            formatted_base_prompt = temp_prompt

        elif chosen_template_idx == 2: # Template expects {length} and possibly {name}
            # Example: "Write a {length} detailed description for this image."
            temp_prompt = selected_prompt_template_str
            if "{name}" in temp_prompt:
                temp_prompt = temp_prompt.replace("{name}", name_to_insert)
            if "{length}" in temp_prompt:
                temp_prompt = temp_prompt.replace("{length}", actual_caption_length_str)
            formatted_base_prompt = temp_prompt
            
        else: # General template (index 0). May contain {name}.
            if "{name}" in selected_prompt_template_str:
                formatted_base_prompt = selected_prompt_template_str.replace("{name}", name_to_insert)
            else:
                formatted_base_prompt = selected_prompt_template_str # No {name} placeholder in this template

    except Exception as e: # Catch any unexpected formatting errors broadly
        print(f"[wo_joycaption_comfyui] Warning: An unexpected error occurred during base prompt formatting for caption_type '{caption_type}', template_idx {chosen_template_idx}. Error: {e}")
        print(f"[wo_joycaption_comfyui] Template was: '{selected_prompt_template_str}'")
        # Fallback: use the template, try to replace {name} at least.
        formatted_base_prompt = selected_prompt_template_str.replace("{name}", name_to_insert)

    # Process extra options
    final_prompt_parts = [formatted_base_prompt]
    if extra_options:
        processed_extra_options = []
        for opt_template in extra_options:
            try:
                # Extra options are expected to only use {name} if they need formatting
                processed_opt = opt_template.replace("{name}", name_to_insert)
                processed_extra_options.append(processed_opt)
            except Exception as e_opt: # Broad catch for safety
                 print(f"[wo_joycaption_comfyui] Warning: Extra option formatting error: '{opt_template}'. Error: {e_opt}")
                 processed_extra_options.append(opt_template) # Add raw option on error
        
        if processed_extra_options:
            final_prompt_parts.append(" ".join(processed_extra_options))

    final_prompt = " ".join(filter(None, final_prompt_parts)) # Join non-empty parts

    # Final checks for unformatted placeholders (these are just warnings)
    # Check against the original selected template to see if a placeholder *should* have been replaced.
    if not name_input and "{NAME}" in final_prompt: # This is expected if name_input was empty
        pass

    if chosen_template_idx == 2 and "{length}" in selected_prompt_template_str and "{length}" in final_prompt:
        print(f"[wo_joycaption_comfyui] Warning: Prompt template for '{caption_type}' might have unformatted '{{length}}'.")

    if chosen_template_idx == 1 and "{word_count}" in selected_prompt_template_str and "{word_count}" in final_prompt:
        print(f"[wo_joycaption_comfyui] Warning: Prompt template for '{caption_type}' might have unformatted '{{word_count}}'.")

    return final_prompt.strip()

class JoyCaptionPredictor:
    def __init__(self, checkpoint_path: str, memory_mode: str, precision, device=None):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = mm.get_torch_device()
        
        self.offload_device = mm.unet_offload_device()
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        
        self.processor = AutoProcessor.from_pretrained(str(checkpoint_path))

        if memory_mode == "Default":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                str(checkpoint_path),
                torch_dtype=self.dtype,
            )
        else:
            from transformers import BitsAndBytesConfig
            qnt_config = BitsAndBytesConfig(
                **MEMORY_EFFICIENT_CONFIGS_DICT[memory_mode],
                # Ensure this key is correct for your config
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                str(checkpoint_path),
                quantization_config=qnt_config
            )
    
    def generate(self, image: Image.Image, system: str, prompt: str, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, seed: int=None, keep_model_loaded: bool=False) -> str:
        # Move model to the target device for generation
        self.model.to(self.device)

        seed_generator = None # Initialize seed_generator
        if seed:
            # Check if CUDA is available and if the target device is a CUDA device
            if torch.cuda.is_available() and "cuda" in str(self.device):
                seed_generator = torch.Generator(device=self.device)
                seed_generator.manual_seed(seed)
            else:
                # Fallback to CPU for seeding if CUDA not available or device is CPU
                seed_generator = torch.Generator(device="cpu") # Explicitly CPU for generator
                seed_generator.manual_seed(seed)

        convo = [
            {
                "role": "system",
                "content": system.strip(),
            },
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ]

        # Format the conversation
        convo_string = self.processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.model.device)
        
        # Ensure pixel_values are on the correct device and dtype
        inputs['pixel_values'] = inputs['pixel_values'].to(device=self.model.device, dtype=self.dtype)

        # Prepare generation arguments
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature > 0 else False,
            "suppress_tokens": None,
            "use_cache": True,
            "temperature": temperature,
            "top_k": None if top_k == 0 else top_k,
            "top_p": top_p,
        }

        # Generate the captions
        generate_ids = self.model.generate(
            **inputs,
            **generation_kwargs
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if not keep_model_loaded:
            self.model.to(self.offload_device)
            mm.soft_empty_cache()

        return caption.strip()


model_directory = os.path.join(folder_paths.models_dir, "LLavacheckpoints")
os.makedirs(model_directory, exist_ok=True)

# https://github.com/kijai/ComfyUI-Florence2/blob/de485b65b3e1b9b887ab494afa236dff4bef9a7e/nodes.py#L36
def create_path_dict(paths: list[str], predicate: Callable[[Path], bool] = lambda _: True) -> dict[str, str]:
    """
    Creates a flat dictionary of the contents of all given paths: ``{name: absolute_path}``.

    Non-recursive.  Optionally takes a predicate to filter items.  Duplicate names overwrite (the last one wins).

    Args:
        paths (list[str]):
            The paths to search for items.
        predicate (Callable[[Path], bool]):
            (Optional) If provided, each path is tested against this filter.
            Returns ``True`` to include a path.

            Default: Include everything
    """
    flattened_paths = [item for path in paths if Path(path).exists() for item in Path(path).iterdir() if predicate(item)]
    return {item.name: str(item.absolute()) for item in flattened_paths}

def get_device_list():
    import torch
    return ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]

def get_comfyui_devices(device_type=None):
    if device_type == "torch":
        return mm.get_torch_device()
    elif device_type == "offload":
        return mm.unet_offload_device()
    else:
        raise ValueError(f"Invalid device_type. Value must be either 'torch' or 'offload'")



class JoyCaptionDownloadAndLoad:
    def __init__(self):
        self.torch_device = get_comfyui_devices(device_type="torch")
        self.offload_device = get_comfyui_devices(device_type="offload")
        
    @classmethod
    def INPUT_TYPES(s):
        models = list(joycaption_config["MODELS"])
        current_devices = get_device_list()
        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "memory_mode": (MEMORY_EFFICIENT_MODES_KEYS, {}),
                "precision_default": (['fp16','bf16','fp32'], {"default": 'fp16'}),
                "device": (
                    current_devices,
                    {"default": current_devices[1] if len(current_devices) > 1 else current_devices[0]}
                ),
            },
        }

    RETURN_TYPES = ("JOYCAPTIONMODEL",)
    RETURN_NAMES = ("joycaption_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "JoyCaption"

    def loadmodel(self, model, memory_mode, precision_default, device):
        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"[wo_joycaption_comfyui] Downloading {model} to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                              local_dir=model_path,
                              local_dir_use_symlinks=False)

        model = JoyCaptionPredictor(model_path, memory_mode, precision_default, device=device)

        return (model,)


class JoyCaptionLoader:
    @classmethod
    def INPUT_TYPES(s):
        all_llm_paths = folder_paths.get_folder_paths("LLavacheckpoints")
        s.model_paths = create_path_dict(all_llm_paths, lambda x: x.is_dir())
        devices = get_device_list()

        return {
            "required": {
                "model": ([*s.model_paths],
                          {"tooltip": "models are expected to be in Comfyui/models/LLavacheckpoints folder"}),
                "memory_mode": (MEMORY_EFFICIENT_MODES_KEYS, {}),
                "precision_default": ([ 'fp16','bf16','fp32'], {"default": 'fp16'}),
                "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0]}),
            },
        }

    RETURN_TYPES = ("JOYCAPTIONMODEL",)
    RETURN_NAMES = ("joycaption_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "JoyCaption"

    def loadmodel(self, model, memory_mode, precision_default, device):
        torch_device = get_comfyui_devices(device_type="torch")
        offload_device = get_comfyui_devices(device_type="offload")
        model_path = JoyCaptionLoader.model_paths.get(model)
        model = JoyCaptionPredictor(model_path, memory_mode, precision_default, device=device)
        return (model,)


class JoyCaption:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "joycaption_model": ("JOYCAPTIONMODEL",),
            "image": ("IMAGE",),
            "caption_type": (CAPTION_TYPE_CHOICES_KEYS, {}),
            "caption_length": (CAPTION_LENGTH_CHOICES_LIST, {"default": "long"}),
            # generation params
            "max_new_tokens":       ("INT",     {"default": 512, "min": 1, "max": 2048}),
            "temperature":          ("FLOAT",   {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "top_p":                ("FLOAT",   {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k":                ("INT",     {"default": 0, "min": 0, "max": 100}),
            "seed":                 ("INT",     {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            "keep_model_loaded":    ("BOOLEAN", {"default": False, "tooltip": "Do not unload model after node execution."}),
        }
        opt = {
            "extra_options": ("EXTRA_OPTION",)
        }
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("query", "caption")
    FUNCTION = "generate"
    CATEGORY = "JoyCaption"

    def generate(self, joycaption_model, image, caption_type, caption_length, max_new_tokens,
                 temperature, top_p, top_k, seed, keep_model_loaded, extra_options=None):
        extras = []
        person_name_from_options = ""
        if extra_options:
            if isinstance(extra_options, tuple) and len(extra_options) == 2:
                extras, person_name_from_options = extra_options
                if not isinstance(extras, list): extras = []
                if not isinstance(person_name_from_options, str): person_name_from_options = ""
            else: # Should not happen if connected to JoyCaptionExtraOptions
                print(f"[wo_joycaption_comfyui] Warning: extra_options is not in the expected format (list, str). Received: {type(extra_options)}")

        prompt = build_prompt(caption_type, caption_length, extras, person_name_from_options)
        system_prompt = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."

        # This is a bit silly. We get the image as a tensor, and we could just use that directly (just need to resize and adjust the normalization).
        # But JoyCaption was trained on images that were resized using lanczos, which I think PyTorch doesn't support.
        # Just to be safe, we'll convert the image to a PIL image and let the processor handle it correctly.
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        response = joycaption_model.generate(
            image=pil_image,
            system=system_prompt,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
        )

        return (prompt, response)


class JoyCaptionCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joycaption_model":     ("JOYCAPTIONMODEL",),
                "image":                ("IMAGE",),
                "system_prompt":        ("STRING",  {"multiline": False, "default": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions." }),
                "user_query":           ("STRING",  {"multiline": True, "default": "Write a detailed description for this image." }),
                # generation params
                "max_new_tokens":       ("INT",     {"default": 512, "min": 1,   "max": 2048}),
                "temperature":          ("FLOAT",   {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p":                ("FLOAT",   {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k":                ("INT",     {"default": 0,   "min": 0,   "max": 100}),
                "seed":                 ("INT",     {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "keep_model_loaded":    ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "JoyCaption"

    def generate(self, joycaption_model, image, system_prompt, user_query, max_new_tokens, temperature, top_p, top_k, seed, keep_model_loaded):
        # This is a bit silly. We get the image as a tensor, and we could just use that directly (just need to resize and adjust the normalization).
        # But JoyCaption was trained on images that were resized using lanczos, which I think PyTorch doesn't support.
        # Just to be safe, we'll convert the image to a PIL image and let the processor handle it correctly.
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        response = joycaption_model.generate(
            image=pil_image,
            system=system_prompt,
            prompt=user_query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
        )

        return (response,)

class JoyCaptionExtraOptions:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {key: ("BOOLEAN", {"default": False, "tooltip": value}) for key, value in OPTIONS_CONFIG_MAP.items()}
        inputs["character_name"] = ("STRING", {"default": "", "multiline": False, "placeholder": "e.g., 'Skywalker'"})
        return {
            "required": inputs
        }

    CATEGORY = "JoyCaption"
    RETURN_TYPES = ("EXTRA_OPTION",)
    RETURN_NAMES = ("extra_options",)
    FUNCTION = "generate_options"

    def generate_options(self, **kwargs):
        selected_options = []
        character_name = None

        for key, value in kwargs.items():
            if key == "character_name":
                character_name = value
            # For each boolean input, if True, append the corresponding string from OPTIONS_CONFIG_MAP
            elif value and key in OPTIONS_CONFIG_MAP:
                selected_options.append(OPTIONS_CONFIG_MAP[key])

        # The return format for ComfyUI: a tuple containing the output value(s).
        # Here, the single output "extra_options" is a tuple of (list_of_strings, character_name_string).
        return ((selected_options, character_name or ""),)


# Optional GGUF support
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    print("[wo_joycaption_comfyui] llama-cpp-python is installed.")

    os.makedirs(os.path.join(folder_paths.models_dir, "llava_gguf"), exist_ok=True)
    os.makedirs(os.path.join(folder_paths.models_dir, "llava_mmproj"), exist_ok=True)

    # llama-cpp is very noisey and spams the console, use a suppressor
    # https://github.com/abetlen/llama-cpp-python/issues/478#issuecomment-1652472173
    import sys
    class suppress_stdout_stderr(object):
        def __enter__(self):
            self.outnull_file = open(os.devnull, 'w')
            self.errnull_file = open(os.devnull, 'w')

            self.old_stdout_fileno_undup = sys.stdout.fileno()
            self.old_stderr_fileno_undup = sys.stderr.fileno()

            self.old_stdout_fileno = os.dup(sys.stdout.fileno())
            self.old_stderr_fileno = os.dup(sys.stderr.fileno())

            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr

            os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
            os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

            sys.stdout = self.outnull_file
            sys.stderr = self.errnull_file
            return self

        def __exit__(self, *_):
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr

            os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
            os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

            os.close(self.old_stdout_fileno)
            os.close(self.old_stderr_fileno)

            self.outnull_file.close()
            self.errnull_file.close()

    # Adapted from https://github.com/without-ordinary/openoutpaint_comfyui_interface/blob/main/py/utils.py
    from io import BytesIO
    import base64
    def image_to_base64_data_uri(image):
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_image}"

    # Partly adapted from: https://github.com/judian17/ComfyUI-joycaption-beta-one-GGUF
    class JoyCaptionPredictorGGUF:
        def __init__(self, gguf_path: str, mmproj_path: str,
                     n_gpu_layers: int = -1, n_ctx: int = 2048,
                     main_gpu: int = 0, split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_NONE):
            self.llm = None
            self.chat_handler_exit_stack = None  # Will store the ExitStack of the chat_handler

            self.gguf_path = gguf_path
            self.mmproj_path = mmproj_path
            self.n_gpu_layers = n_gpu_layers
            self.n_ctx = n_ctx
            self.main_gpu = main_gpu
            self.split_mode = split_mode

        def _load_model(self):
            try:
                chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)
                if hasattr(chat_handler, '_exit_stack'):
                    self.chat_handler_exit_stack = chat_handler._exit_stack
                else:
                    print("[wo_joycaption_comfyui] Warning: Llava15ChatHandler does not have _exit_stack attribute.")

                self.llm = Llama(
                    model_path=self.gguf_path,
                    chat_handler=chat_handler,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    main_gpu=self.main_gpu,
                    split_mode=self.split_mode,
                    verbose=False,
                )
            except Exception as e:
                print(f"[wo_joycaption_comfyui]: Error loading GGUF model: {e}")
                self._unload_model()  # Unload to be safe?
                raise e

        def _unload_model(self):
            if self.chat_handler_exit_stack is not None:
                try:
                    self.chat_handler_exit_stack.close()
                except Exception as e_close:
                    print(f"[wo_joycaption_comfyui]: Error closing chat_handler_exit_stack (unload_after_generate): {e_close}")
                self.chat_handler_exit_stack = None

            if self.llm is not None:
                del self.llm
                self.llm = None  # Explicitly set to None
                mm.soft_empty_cache()

        @torch.inference_mode()
        def generate(self, image: Image.Image, system: str, prompt: str, max_new_tokens: int, temperature: float,
                     top_p: float, top_k: int, seed: int = None, keep_model_loaded: bool = False) -> str:
            # load model if needed
            if self.llm is None: self._load_model()

            if seed: self.llm.set_seed(seed)

            convo = [
                {
                    "role": "system",
                    "content": system.strip(),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_base64_data_uri(image)},
                        },
                        {
                            "type": "text",
                            "content": prompt.strip(),
                        },
                    ]
                }
            ]

            with suppress_stdout_stderr():
                response = self.llm.create_chat_completion(
                    messages=convo,
                    max_tokens=max_new_tokens if max_new_tokens > 0 else None,
                    temperature=temperature if temperature > 0 else 0.0,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else 0,
                )
                caption = response['choices'][0]['message']['content']

            if not keep_model_loaded:
                self._unload_model()

            return caption.strip()

    LLAMA_SPLIT_MODES = {
        "LLAMA_SPLIT_MODE_NONE": llama_cpp.LLAMA_SPLIT_MODE_NONE,
        "LLAMA_SPLIT_MODE_ROW": llama_cpp.LLAMA_SPLIT_MODE_ROW,
        "LLAMA_SPLIT_MODE_LAYER": llama_cpp.LLAMA_SPLIT_MODE_LAYER,
    }

    class JoyCaptionLoaderGGUF:
        @classmethod
        def INPUT_TYPES(s):
            devices = [f"{i}" for i in range(torch.cuda.device_count())]
            split_modes = list(LLAMA_SPLIT_MODES.keys())
            return {
                "required": {
                    "gguf_model": (folder_paths.get_filename_list("llava_gguf"), ),
                    "mmproj_file": (folder_paths.get_filename_list("llava_mmproj"), ),
                    "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000}),
                    "n_ctx": ("INT", {"default": 2048, "min": 512, "max": 8192}),
                    "main_gpu": (devices, {"default": devices[0]}),
                    "split_mode": (split_modes, {"default": split_modes[0]}),
                },
            }

        RETURN_TYPES = ("JOYCAPTIONMODEL",)
        RETURN_NAMES = ("joycaption_model",)
        FUNCTION = "loadmodel"
        CATEGORY = "JoyCaption"

        def loadmodel(self, gguf_model, mmproj_file, n_gpu_layers, n_ctx, main_gpu, split_mode):
            gguf_path = folder_paths.get_full_path_or_raise("llava_gguf", gguf_model)
            mmproj_path = folder_paths.get_full_path_or_raise("llava_mmproj", mmproj_file)
            model = JoyCaptionPredictorGGUF(
                gguf_path,
                mmproj_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                main_gpu=int(main_gpu),
                split_mode=LLAMA_SPLIT_MODES[split_mode],
            )
            return (model,)

except ImportError:
    print("[wo_joycaption_comfyui] llama-cpp-python required for GGUF support is not installed. JoyCaption GGUF node will be missing.")
except Exception as e:
    print("[wo_joycaption_comfyui] failed to load GGUF support: {e}")
