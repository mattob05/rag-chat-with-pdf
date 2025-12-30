import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)
from langchain_huggingface import HuggingFacePipeline

MODEL_NAME = 'microsoft/Phi-3-mini-4k-instruct'

class LLMEngine:
    def __init__(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=self.bnb_config,
            device_map={"": 0},
        )

    def get_llm(self):
        pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
            num_return_sequences=1,
        )

        return HuggingFacePipeline(pipeline=pipe)