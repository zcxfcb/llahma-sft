from unsloth import FastLanguageModel
import os
from transformers import TextStreamer

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True 

instruction = "Generate code to parse an input SQL string."
input = "[1, 2, 3, 4, 5]"
huggingface_model_name = "zcx1111/Llama-3.1-8B-bnb-4bit-python"

# 2. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = os.getenv("HF_TOKEN")
)

# 3. Inference and print
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    instruction
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2046)