import os.path
from os import mkdir
from modules import paths, shared
from modules.sd_models import model_path as sd_model_dir_path
from basicsr.utils.download_util import load_file_from_url


def LoadModel(url, dir, name, path):
    if not os.path.exists(path):
        if not os.path.exists(dir):
            os.makedirs(dir)

        load_file_from_url(url, dir, True, name)


def CheckModelsExist():
    print("Witchpot initialization")

    # StableDiffusion
    sd_model_url = ""
    #sd_model_dir_path = os.path.abspath(os.path.join(paths.models_path, "Stable-diffusion"))
    sd_model_name = ""
    sd_model_path = os.path.abspath(os.path.join(sd_model_dir_path, sd_model_name))
    
    print("StableDiffusion_dir : " + sd_model_dir_path)        
    #LoadModel(sd_model_url, sd_model_dir_path, sd_model_name, sd_model_path)

    # LoRA
    lora_model_url = "https://huggingface.co/Witchpot/icestage/resolve/main/witchpot-icestage-sd-1-5.safetensors"
    lora_models_dir_path = os.path.abspath(shared.cmd_opts.lora_dir)
    lora_model_name = "witchpot-icestage-sd-1-5.safetensors"
    lora_model_path = os.path.abspath(os.path.join(lora_models_dir_path, lora_model_name))

    print("LoRA_dir : " + lora_models_dir_path)
    LoadModel(lora_model_url, lora_models_dir_path, lora_model_name, lora_model_path)

    # ControlNet
    cn_model_url = "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors"
    cn_models_dir_path = os.path.abspath(os.path.join(paths.models_path, "ControlNet"))
    cn_model_name = "control_v11f1p_sd15_depth_fp16.safetensors"
    cn_model_path = os.path.abspath(os.path.join(cn_models_dir_path, cn_model_name))

    print("ControlNet_dir : " + cn_models_dir_path)
    LoadModel(cn_model_url, cn_models_dir_path, cn_model_name, cn_model_path)