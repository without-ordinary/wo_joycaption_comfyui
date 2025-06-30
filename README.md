# JoyCaption ComfyUI Nodes

Fork of [fpgaminer's nodes](https://github.com/fpgaminer/joycaption_comfyui) with substantial changes to fit our use cases.

## Usage

The `JoyCaption` node allows you to pick the desired Caption Type, Caption Length, and various additional Extra Options for tweaking the captions.  It takes an Image as input, and will output the Query that was built and sent to the JoyCaption model based on your settings, and the resulting Caption.  This node automatically downloads the JoyCaption model into `LLavacheckpoints`.  The model can be loaded in 8-bit or 4-bit quantization if low memory usage is needed (though quality will degrade).

The `JoyCaption (Custom)` node works much like the `JoyCaption` node but lets you input a custom System Prompt and Query.

![Screenshot of example workflow](https://raw.githubusercontent.com/without-ordinary/wo_joycaption_comfyui/refs/heads/main/workflow_example.png)

## GGUF Support (optional)

Requires [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/) with cuda support.

The GGUF loader node will be available if `llama-cpp-python` is installed.