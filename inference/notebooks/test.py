import os
from glob import glob
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# 1) load your model & processor as before
model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 2) robust frames→video function
def frames_to_video(frame_folder, out_path, fps=1):
    # accept lots of extensions
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tiff")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(frame_folder, ext)))
    files = sorted(files)
    if not files:
        raise ValueError(f"No images found in {frame_folder!r}")
    
    # use PIL to get size (W, H)
    first = Image.open(files[0]).convert("RGB")
    w, h = first.size
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    for img_path in files:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")
            continue
        arr = np.array(img)             # H×W×3 RGB
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

# 3) run it
frames_folder    =  "/code/Datasets/alireza_projects/video_saliency/visal/datasets/dataset_resize/.../train/clip1/images"
temp_video_path  = "temp_from_frames.mp4"
frames_to_video(frames_folder, temp_video_path, fps=25)

# 4) build & run exactly your original code:
conversation = [
    {
        "role": "user",
        "content": [
            {   "type": "video",
                "video": {
                    "video_path": temp_video_path,
                    "fps": 1,
                    "max_frames": 128
                },
            },
            {"type": "text", "text": "Describe the video with detail"},
        ],
    }
]

inputs = processor(conversation=conversation, return_tensors="pt")
inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

output_ids = model.generate(**inputs, max_new_tokens=256)
response   = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(response)
