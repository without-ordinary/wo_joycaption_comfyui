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
import folder_paths


# From (https://github.com/gokayfem/ComfyUI_VLM_nodes/blob/1ca496c1c8e8ada94d7d2644b8a7d4b3dc9729b3/nodes/qwen2vl.py)
# Apache 2.0 License



def read_joycaption_config(file_path):
  """读取 JSON 文件并转换为 Python 字典。"""
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    return data
  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    return None
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{file_path}'.")
    return None

joycaption_node_path = os.path.dirname(os.path.realpath(__file__))
joycaption_config = read_joycaption_config(os.path.join(joycaption_node_path, "joycaption_config.json"))

# Store the actual data structures from the config
CAPTION_TYPE_CONFIG_MAP = joycaption_config["CAPTION_TYPE_MAP"]  # This is a dictionary
CAPTION_LENGTH_CHOICES_LIST = list(joycaption_config["CAPTION_LENGTH_CHOICES"]) # This is a list of strings
MEMORY_EFFICIENT_CONFIGS_DICT = joycaption_config["MEMORY_EFFICIENT_CONFIGS"] # This is a dictionary

# Derive lists of keys/choices specifically for UI elements (dropdowns)
CAPTION_TYPE_CHOICES_KEYS = list(CAPTION_TYPE_CONFIG_MAP.keys())
# CAPTION_LENGTH_CHOICES_LIST is already suitable for UI choices
MEMORY_EFFICIENT_MODES_KEYS = list(MEMORY_EFFICIENT_CONFIGS_DICT.keys())

def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
    prompt_templates_list = []

    if caption_type not in CAPTION_TYPE_CONFIG_MAP:
        print(f"JoyCaption Warning: Unknown caption_type '{caption_type}'. Attempting to use default.")
        if not CAPTION_TYPE_CONFIG_MAP or not list(CAPTION_TYPE_CONFIG_MAP.keys()):
            print(f"JoyCaption Error: CAPTION_TYPE_CONFIG_MAP is empty or invalid. Cannot determine default prompt.")
            return "Error: CAPTION_TYPE_CONFIG_MAP is misconfigured."
        
        default_template_key = list(CAPTION_TYPE_CONFIG_MAP.keys())[0]
        print(f"JoyCaption Warning: Using default caption type '{default_template_key}'.")
        prompt_templates_list = CAPTION_TYPE_CONFIG_MAP.get(default_template_key, [])
        
        if not prompt_templates_list:
             print(f"JoyCaption Error: Default caption type '{default_template_key}' has no templates.")
             return f"Error: No templates for default type {default_template_key}."
    else:
        prompt_templates_list = CAPTION_TYPE_CONFIG_MAP.get(caption_type, [])
        if not prompt_templates_list:
            print(f"JoyCaption Error: Caption type '{caption_type}' has no templates defined.")
            return f"Error: No templates for caption type '{caption_type}'."

    # Determine which template to use from the list based on caption_length
    # Template indices: 0 for "any", 1 for "{word_count}", 2 for "{length}"
    chosen_template_idx = 0 
    actual_caption_length_str = str(caption_length) 

    if actual_caption_length_str.isdigit():
        if len(prompt_templates_list) > 1:
            chosen_template_idx = 1  # Use template with {word_count}
        else:
            print(f"JoyCaption Warning: Not enough templates for '{caption_type}' to use specific word count. Using general template (index 0).")
            chosen_template_idx = 0
    elif actual_caption_length_str != "any":  # Descriptive length like "short", "long"
        if len(prompt_templates_list) > 2:
            chosen_template_idx = 2  # Use template with {length}
        else:
            print(f"JoyCaption Warning: Not enough templates for '{caption_type}' to use descriptive length. Using general template (index 0).")
            chosen_template_idx = 0
    
    if chosen_template_idx >= len(prompt_templates_list):
        print(f"JoyCaption Warning: Template index {chosen_template_idx} out of bounds for '{caption_type}' (list size {len(prompt_templates_list)}). Falling back to index 0.")
        chosen_template_idx = 0
        if not prompt_templates_list : 
            print(f"JoyCaption Error: Critical - no templates available for {caption_type} after fallback.")
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
        print(f"JoyCaption Warning: An unexpected error occurred during base prompt formatting for caption_type '{caption_type}', template_idx {chosen_template_idx}. Error: {e}")
        print(f"Template was: '{selected_prompt_template_str}'")
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
                 print(f"JoyCaption Warning: Extra option formatting error: '{opt_template}'. Error: {e_opt}")
                 processed_extra_options.append(opt_template) # Add raw option on error
        
        if processed_extra_options:
            final_prompt_parts.append(" ".join(processed_extra_options))

    final_prompt = " ".join(filter(None, final_prompt_parts)) # Join non-empty parts

    # Final checks for unformatted placeholders (these are just warnings)
    # Check against the original selected template to see if a placeholder *should* have been replaced.
    if not name_input and "{NAME}" in final_prompt: # This is expected if name_input was empty
        pass
    
    if chosen_template_idx == 2 and "{length}" in selected_prompt_template_str and "{length}" in final_prompt:
        print(f"JoyCaption (GGUF) Warning: Prompt template for '{caption_type}' might have unformatted '{{length}}'.")
    
    if chosen_template_idx == 1 and "{word_count}" in selected_prompt_template_str and "{word_count}" in final_prompt:
        print(f"JoyCaption (GGUF) Warning: Prompt template for '{caption_type}' might have unformatted '{{word_count}}'.")

    return final_prompt.strip()

class JoyCaptionPredictor:
    def __init__(self, checkpoint_path: str, memory_mode: str, precision, device=None):
        if device is not None:
            self.device = torch.device(device) # Store as instance attribute
        else:
            self.device = mm.get_torch_device() # Store as instance attribute
        
        self.offload_device = mm.unet_offload_device() # Store as instance attribute
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision] # Store as instance attribute
        
        self.processor = AutoProcessor.from_pretrained(str(checkpoint_path))

        if memory_mode == "Default":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                str(checkpoint_path),
                torch_dtype=self.dtype, # Use self.dtype
            )#.to(self.device) # Moving to device can be done here or explicitly in generate
        else:
            from transformers import BitsAndBytesConfig
            qnt_config = BitsAndBytesConfig(
                **MEMORY_EFFICIENT_CONFIGS_DICT[memory_mode],
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"], # Ensure this key is correct for your config
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                str(checkpoint_path),
                quantization_config=qnt_config
            )#.to(self.device) # Moving to device can be done here or explicitly in generate
            # output_dir = os.path.join(checkpoint_path, "pre-quantized")
            # self.model.save_pretrained(output_dir, safe_serialization=True)
            # self.processor.save_pretrained(output_dir)
        print(f"Loaded model {checkpoint_path} with memory mode {memory_mode}")
        # Consider moving the model to self.device here if it should always reside on it
        # self.model.to(self.device) 
    
    def generate(self, image: Image.Image, system: str, prompt: str, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, seed: int=None, keep_model_loaded: bool=False) -> str:
        # Move model to the target device for generation
        self.model.to(self.device) # Use self.device

        seed_generator = None # Initialize seed_generator
        if seed:
            # Check if CUDA is available and if the target device is a CUDA device
            if torch.cuda.is_available() and "cuda" in str(self.device): # Use self.device
                print(f"cuda seed: {seed}")
                seed_generator = torch.Generator(device=self.device) # Use self.device
                seed_generator.manual_seed(seed)
            else:
                # Fallback to CPU for seeding if CUDA not available or device is CPU
                print(f"cpu seed: {seed}")
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
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.model.device) # Use self.model.device
        
        # Ensure pixel_values are on the correct device and dtype
        inputs['pixel_values'] = inputs['pixel_values'].to(device=self.model.device, dtype=self.dtype) # Use self.dtype and self.model.device

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
            self.model.to(self.offload_device) # Use self.offload_device
            mm.soft_empty_cache()

        return caption.strip()


model_directory = os.path.join(folder_paths.models_dir, "LLavacheckpoints")
os.makedirs(model_directory, exist_ok=True)

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
        # memory_modes = list(joycaption_config["MEMORY_EFFICIENT_CONFIGS"].keys())
        # models = list(joycaption_config["MODELS"])
        current_devices = get_device_list() # Define devices here
        return {
            "required": {
                "model": (list(joycaption_config["MODELS"]), # Assuming MODELS in JSON is a list
                          {"default": 'fancyfeast/llama-joycaption-beta-one-hf-llava'}),
                "memory_mode": (MEMORY_EFFICIENT_MODES_KEYS, {}), # Corrected
                "precision_default": ([ 'fp16','bf16','fp32'], {"default": 'fp16'}),
                "device": (current_devices, {"default": current_devices[1] if len(current_devices) > 1 else current_devices[0]}), # Fixed NameError
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
            print(f"Downloading {model} to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                              local_dir=model_path,
                              local_dir_use_symlinks=False)

        model = JoyCaptionPredictor(model_path, memory_mode, precision_default, device=device)

        return (model,)


class JoyCaptionLoader:
    @classmethod
    def INPUT_TYPES(s):
        # memory_modes = list(joycaption_config["MEMORY_EFFICIENT_CONFIGS"].keys())
        all_llm_paths = folder_paths.get_folder_paths("LLavacheckpoints")
        s.model_paths = create_path_dict(all_llm_paths, lambda x: x.is_dir())
        devices = get_device_list()

        return {
            "required": {
                "model": ([*s.model_paths],
                          {"tooltip": "models are expected to be in Comfyui/models/LLavacheckpoints folder"}),
                "memory_mode": (MEMORY_EFFICIENT_MODES_KEYS, {}), # Corrected
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
        print(f"Loading model from {model_path}")
        model = JoyCaptionPredictor(model_path, memory_mode, precision_default, device=device)
        return (model,)


class JoyCaption:
    @classmethod
    def INPUT_TYPES(cls):
        # caption_lengths = list(joycaption_config["CAPTION_LENGTH_CHOICES"]) # Old
        # caption_types = list(joycaption_config["CAPTION_TYPE_MAP"].keys())   # Old
        req = {
            "joycaption_model": ("JOYCAPTIONMODEL",),
            "image": ("IMAGE",),
            "caption_type": (CAPTION_TYPE_CHOICES_KEYS, {}), # Corrected
            "caption_length": (CAPTION_LENGTH_CHOICES_LIST, {"default": "long"}), # Corrected
            # "extra_option1": (list(EXTRA_OPTIONS),),
            # "extra_option2": (list(EXTRA_OPTIONS),),
            # "extra_option3": (list(EXTRA_OPTIONS),),
            # "extra_option4": (list(EXTRA_OPTIONS),),
            # "extra_option5": (list(EXTRA_OPTIONS),),
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
                print(f"JoyCaption Warning: extra_options is not in the expected format (list, str). Received: {type(extra_options)}")

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
        # ... (INPUT_TYPES definition remains the same)
        return {
            "required": {
                "refer_character_name": ("BOOLEAN", {"default": False}), "exclude_people_info": ("BOOLEAN", {"default": False}), "include_lighting": ("BOOLEAN", {"default": False}),
                "include_camera_angle": ("BOOLEAN", {"default": False}), "include_watermark_info": ("BOOLEAN", {"default": False}), "include_JPEG_artifacts": ("BOOLEAN", {"default": False}),
                "include_exif": ("BOOLEAN", {"default": False}), "exclude_sexual": ("BOOLEAN", {"default": False}), "exclude_image_resolution": ("BOOLEAN", {"default": False}),
                "include_aesthetic_quality": ("BOOLEAN", {"default": False}), "include_composition_style": ("BOOLEAN", {"default": False}), "exclude_text": ("BOOLEAN", {"default": False}),
                "specify_depth_field": ("BOOLEAN", {"default": False}), "specify_lighting_sources": ("BOOLEAN", {"default": False}), "do_not_use_ambiguous_language": ("BOOLEAN", {"default": False}),
                "include_nsfw_rating": ("BOOLEAN", {"default": False}), "only_describe_most_important_elements": ("BOOLEAN", {"default": False}), "do_not_include_artist_name_or_title": ("BOOLEAN", {"default": False}),
                "identify_image_orientation": ("BOOLEAN", {"default": False}), "use_vulgar_slang_and_profanity": ("BOOLEAN", {"default": False}), "do_not_use_polite_euphemisms": ("BOOLEAN", {"default": False}),
                "include_character_age": ("BOOLEAN", {"default": False}), "include_camera_shot_type": ("BOOLEAN", {"default": False}), "exclude_mood_feeling": ("BOOLEAN", {"default": False}),
                "include_camera_vantage_height": ("BOOLEAN", {"default": False}), "mention_watermark_explicitly": ("BOOLEAN", {"default": False}), "avoid_meta_descriptive_phrases": ("BOOLEAN", {"default": False}),
                "character_name": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g., 'Skywalker'"}),
            }
        }

    CATEGORY = "JoyCaption"
    RETURN_TYPES = ("EXTRA_OPTION",)
    RETURN_NAMES = ("extra_options",)
    FUNCTION = "generate_options"

    def generate_options(self, refer_character_name, exclude_people_info, include_lighting, include_camera_angle,
                         include_watermark_info, include_JPEG_artifacts, include_exif, exclude_sexual,
                         exclude_image_resolution, include_aesthetic_quality, include_composition_style,
                         exclude_text, specify_depth_field, specify_lighting_sources,
                         do_not_use_ambiguous_language, include_nsfw_rating, only_describe_most_important_elements,
                         do_not_include_artist_name_or_title, identify_image_orientation, use_vulgar_slang_and_profanity,
                         do_not_use_polite_euphemisms, include_character_age, include_camera_shot_type,
                         exclude_mood_feeling, include_camera_vantage_height, mention_watermark_explicitly,
                         avoid_meta_descriptive_phrases, character_name):

        # Access the EXTRA_MAP dictionary directly from the loaded joycaption_config
        options_config_map = joycaption_config["EXTRA_MAP"]

        selected_options = []
        
        # For each boolean input, if True, append the corresponding string from options_config_map
        if refer_character_name: selected_options.append(options_config_map["refer_character_name"])
        if exclude_people_info: selected_options.append(options_config_map["exclude_people_info"])
        if include_lighting: selected_options.append(options_config_map["include_lighting"])
        if include_camera_angle: selected_options.append(options_config_map["include_camera_angle"])
        if include_watermark_info: selected_options.append(options_config_map["include_watermark_info"])
        if include_JPEG_artifacts: selected_options.append(options_config_map["include_JPEG_artifacts"])
        if include_exif: selected_options.append(options_config_map["include_exif"]) # Corrected line
        if exclude_sexual: selected_options.append(options_config_map["exclude_sexual"])
        if exclude_image_resolution: selected_options.append(options_config_map["exclude_image_resolution"])
        if include_aesthetic_quality: selected_options.append(options_config_map["include_aesthetic_quality"])
        if include_composition_style: selected_options.append(options_config_map["include_composition_style"])
        if exclude_text: selected_options.append(options_config_map["exclude_text"])
        if specify_depth_field: selected_options.append(options_config_map["specify_depth_field"])
        if specify_lighting_sources: selected_options.append(options_config_map["specify_lighting_sources"])
        if do_not_use_ambiguous_language: selected_options.append(options_config_map["do_not_use_ambiguous_language"])
        if include_nsfw_rating: selected_options.append(options_config_map["include_nsfw_rating"])
        if only_describe_most_important_elements: selected_options.append(options_config_map["only_describe_most_important_elements"])
        if do_not_include_artist_name_or_title: selected_options.append(options_config_map["do_not_include_artist_name_or_title"])
        if identify_image_orientation: selected_options.append(options_config_map["identify_image_orientation"])
        if use_vulgar_slang_and_profanity: selected_options.append(options_config_map["use_vulgar_slang_and_profanity"])
        if do_not_use_polite_euphemisms: selected_options.append(options_config_map["do_not_use_polite_euphemisms"])
        if include_character_age: selected_options.append(options_config_map["include_character_age"])
        if include_camera_shot_type: selected_options.append(options_config_map["include_camera_shot_type"])
        if exclude_mood_feeling: selected_options.append(options_config_map["exclude_mood_feeling"])
        if include_camera_vantage_height: selected_options.append(options_config_map["include_camera_vantage_height"])
        if mention_watermark_explicitly: selected_options.append(options_config_map["mention_watermark_explicitly"])
        if avoid_meta_descriptive_phrases: selected_options.append(options_config_map["avoid_meta_descriptive_phrases"])

        # The return format for ComfyUI: a tuple containing the output value(s).
        # Here, the single output "extra_options" is a tuple of (list_of_strings, character_name_string).
        return ((selected_options, character_name or ""),)
