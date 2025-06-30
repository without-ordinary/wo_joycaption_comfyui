from . import nodes
import os
import folder_paths

folder_paths.add_model_folder_path("LLavacheckpoints", os.path.join(folder_paths.models_dir, "LLavacheckpoints"), is_default=True)
folder_paths.add_model_folder_path("llava_gguf", os.path.join(folder_paths.models_dir, "llava_gguf"), is_default=True)
folder_paths.add_model_folder_path("llava_mmproj", os.path.join(folder_paths.models_dir, "llava_mmproj"), is_default=True)


NODE_CLASS_MAPPINGS = {
    "WO_JoyCaption": nodes.JoyCaption,
    "WO_JoyCaption_Custom": nodes.JoyCaptionCustom,
    "WO_JoyCaption_DownloadAndLoad": nodes.JoyCaptionDownloadAndLoad,
    "WO_JoyCaption_Loader": nodes.JoyCaptionLoader,
    "WO_JoyCaption_ExtraOptions": nodes.JoyCaptionExtraOptions,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WO_JoyCaption": "JoyCaption",
    "WO_JoyCaption_Custom": "JoyCaption (Custom)",
    "WO_JoyCaption_DownloadAndLoad": "JoyCaption Download And Load",
    "WO_JoyCaption_Loader": "JoyCaption Loader",
    "WO_JoyCaption_ExtraOptions": "JoyCaption Extra Options",
}

#
if nodes.JoyCaptionLoaderGGUF:
    print("=== nodes.JoyCaptionLoaderGGUF ===")
    NODE_CLASS_MAPPINGS["WO_JoyCaption_GGUF_Loader"] = nodes.JoyCaptionLoaderGGUF
    NODE_DISPLAY_NAME_MAPPINGS["WO_JoyCaption_GGUF_Loader"] = "JoyCaption GGUF Loader"
else:
    print("=== no nodes.JoyCaptionLoaderGGUF ===")