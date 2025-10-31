# local_llm.py
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_qwen_llm():
    model_path = "./model/Qwen2.5-0.5B-Instruct"
    print('loading qwen model...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )

    print('创建pipeline')

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    print('pipeline创建完成')

    return HuggingFacePipeline(pipeline=pipe)
