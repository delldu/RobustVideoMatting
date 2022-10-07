"""Video Matte Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import redos
import todos

from . import matte

import pdb


def get_model():
    """Create model."""

    device = todos.model.get_device()

    model_path = "models/video_matte.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = matte.MattingNetwork(backbone="mobilenetv3")
    todos.model.load(model, checkpoint)
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/video_matte.torch"):
        model.save("output/video_matte.torch")

    return model, device


def model_forward(model, device, input_tensor, multi_times=1):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    # convert tensor from Bx4xHxW to Bx3xHxW
    output_tensor = todos.model.forward(model, device, input_tensor)

    return output_tensor[:, :, 0:H, 0:W]


def video_predict(input_file, output_file):
    # Load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # Load model
    model, device = get_model()

    print(f"  matte {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def matte_video_frame(no, data):
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        if no == 0: # First repeat for stable first hidden state !!!
            for i in range(5): 
                model_forward(model, device, input_tensor[:, 0:3, :, :])

        output_tensor = model_forward(model, device, input_tensor[:, 0:3, :, :])

        temp_output_file = f"{output_dir}/{no + 1:06d}.png"
        # todos.data.save_tensor(output_tensor, temp_output_file)
        todos.data.save_tensor([input_tensor, output_tensor], temp_output_file)

    video.forward(callback=matte_video_frame)

    redos.video.encode(output_dir, output_file)

    # Delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()

    return True