import requests
import torch
import os
import io
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen
import time

# Set max_split_size_mb to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
    # if you do not use Ampere or later GPUs, change attention to "eager"
    _attn_implementation='flash_attention_2',
).cuda()

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path)

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

prompt = """
６つの画像から文字を抽出してください。

以下の6枚の画像は1つの縦長画像を6分割したものです。
画像の配置は次のようになっています：
1 | 2
3 | 4
5 | 6

今回の使用する画像は1枚目です。
また、日本語を正しく認識するように jpn 言語モデルを使用し、句読点や改行をできるだけ正確に保持してください。

出力するテキストは、
* 画像の順序を保ちつつ、
* 重複部分を自動で削除し、
* 文章の流れが自然になるように統合してください。
"""

prompt = f'{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

# Measure processing time
start = time.perf_counter()

# Download and open images
path1 = '/home/user/phi-4-server/data/sub_image_1.png'
path2 = '/home/user/phi-4-server/data/sub_image_2.png'
image1 = Image.open(path1)
# image2 = Image.open(path2)
inputs = processor(text=prompt, images=[image1], return_tensors='pt').to('cuda:0')

# Generate response
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)

# Clean up to free memory
del image1
torch.cuda.empty_cache()

generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')
print(f'処理時間: {time.perf_counter() - start:.2f}秒')