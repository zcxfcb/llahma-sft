from unsloth import FastLanguageModel
import os
from transformers import TextStreamer

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True 
huggingface_model_name = "zcx1111/Llama-3.1-8B-bnb-4bit-python"
local_model_dir = "model"

# 2. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "local_model_dir",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = os.getenv("HF_TOKEN")
)

# 3. Upload Model
model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))

# # Merge to 4bit
# if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "merged_4bit", token = os.getenv("HF_TOKEN"))

# # Just LoRA adapters
# if True: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
# if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "lora", token = os.getenv("HF_TOKEN"))

# # Save to 8bit Q8_0
# if True: model.save_pretrained_gguf("model", tokenizer,)
# if True: model.push_to_hub_gguf(huggingface_model_name, tokenizer, token = os.getenv("HF_TOKEN"))

# # Save to 16bit GGUF
# if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if True: model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "f16", token = os.getenv("HF_TOKEN"))

# # Save to q4_k_m GGUF
# if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
# if True: model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "q4_k_m", token = os.getenv("HF_TOKEN"))

# # Save to multiple GGUF options - much faster if you want multiple!
# if True:
#     model.push_to_hub_gguf(
#         huggingface_model_name, # Change hf to your username!
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
#         token = os.getenv("HF_TOKEN")
#     )