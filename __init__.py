from . import nodes
import os
import folder_paths

folder_paths.add_model_folder_path("LLavacheckpoints", os.path.join(folder_paths.models_dir, "LLavacheckpoints"), is_default=True)


NODE_CLASS_MAPPINGS = {
	"JJC_JoyCaption": nodes.JoyCaption,
	"JJC_JoyCaption_Custom": nodes.JoyCaptionCustom,
	"JJC_JoyCaption_DownloadAndLoad": nodes.JoyCaptionDownloadAndLoad,
	"JJC_JoyCaption_Loader": nodes.JoyCaptionLoader,
    "JJC_JoyCaption_ExtraOptions": nodes.JoyCaptionExtraOptions,
}
NODE_DISPLAY_NAME_MAPPINGS = {
	"JJC_JoyCaption": "JoyCaption",
	"JJC_JoyCaption_Custom": "JoyCaption (Custom)",
	"JJC_JoyCaption_DownloadAndLoad": "JoyCaption Download And Load",
	"JJC_JoyCaption_Loader": "JoyCaption Loader",
    "JJC_JoyCaption_ExtraOptions": "JoyCaption Extra Options",
}
