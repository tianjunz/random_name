# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# from fastchat.train.llama_xformer_monkey_patch import replace_llama_attn_with_xformer

replace_llama_attn_with_flash_attn()
# replace_llama_attn_with_xformer()

from fastchat.train.train_rag1 import train_rag

if __name__ == "__main__":
    train_rag()
