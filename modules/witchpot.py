import os.path
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
    print("StableDiffusion_dir : " + sd_model_dir_path)

    #sd_model_url = ""
    #sd_model_name = ""
    #sd_model_path = os.path.abspath(os.path.join(sd_model_dir_path, sd_model_name))
    #LoadModel(sd_model_url, sd_model_dir_path, sd_model_name, sd_model_path)

    # LoRA
    lora_models_dir_path = os.path.abspath(shared.cmd_opts.lora_dir)
    print("LoRA_dir : " + lora_models_dir_path)

    lora_model_url = "https://huggingface.co/Witchpot/icestage/resolve/main/witchpot-icestage-sd-1-5.safetensors"
    lora_model_name = "witchpot-icestage-sd-1-5.safetensors"
    lora_model_path = os.path.abspath(os.path.join(lora_models_dir_path, lora_model_name))
    LoadModel(lora_model_url, lora_models_dir_path, lora_model_name, lora_model_path)

    lora_model_url = "https://huggingface.co/Witchpot/IsometricCanalCity/resolve/main/witchpot_isometric_canal_city.safetensors"
    lora_model_name = "witchpot_isometric_canal_city.safetensors"
    lora_model_path = os.path.abspath(os.path.join(lora_models_dir_path, lora_model_name))
    LoadModel(lora_model_url, lora_models_dir_path, lora_model_name, lora_model_path)

    # ControlNet
    cn_models_dir_path = os.path.abspath(os.path.join(paths.models_path, "ControlNet"))
    print("ControlNet_dir : " + cn_models_dir_path)

    cn_depth_model_url = "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors"
    cn_depth_model_name = "control_v11f1p_sd15_depth_fp16.safetensors"
    cn_depth_model_path = os.path.abspath(os.path.join(cn_models_dir_path, cn_depth_model_name))
    LoadModel(cn_depth_model_url, cn_models_dir_path, cn_depth_model_name, cn_depth_model_path)

    cn_normal_model_url = "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors"
    cn_normal_model_name = "control_v11p_sd15_normalbae_fp16.safetensors"
    cn_normal_model_path = os.path.abspath(os.path.join(cn_models_dir_path, cn_normal_model_name))
    LoadModel(cn_normal_model_url, cn_models_dir_path, cn_normal_model_name, cn_normal_model_path)
