import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openvino_diarize import OpenVINOSpeechDiarizeLoader


def initialize_shared(q):
    global queue
    queue=q

def loadWordLevelASR(model_id, asr_device):
    print("Load ASR MODEL", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if "GPU" == asr_device else torch.float32,  
        #low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    print("loading processor", flush=True)

    #model.to('xpu' if args.device == "GPU" else args.device)
    #print("ASR device model: ", model.device)
    processor = AutoProcessor.from_pretrained(model_id)
    print("loading pipeline", flush=True)
    asr_loader = pipeline(
        "automatic-speech-recognition",
        model=model.to('xpu') if "GPU" == asr_device else model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device='xpu' if asr_device == 'GPU' else asr_device,
    )
    return asr_loader

def loadDiarizer(endpoint, audio_file, model_id):
    diarize_loader = OpenVINOSpeechDiarizeLoader(
        api_base = None if endpoint == "" else endpoint,
        file_path = audio_file,
        model_id = model_id
    )
    return diarize_loader

def loadSentimenter(endpoint, sentiment_model_id, device, batch_size, max_tokens):
    if endpoint == "":
        print("Loading local sentiment...")
        ov_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "CACHE_DIR": "./cache-ov-model",
        }
        ov_llm = HuggingFacePipeline.from_model_id(
            model_id=sentiment_model_id,
            device="cpu",
            task="text-generation",
            backend="openvino",
            batch_size=batch_size,
            model_kwargs={
                "device": device,
                "ov_config": ov_config
            },
            pipeline_kwargs={
                "max_new_tokens": max_tokens,
                "return_full_text": False,
    #            "repetition_penalty": 1.2,
    #            "encoder_repetition_penalty": 1.2,
    #            "top_p": 0.8,
                "temperature": 0.2,
            })
        ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id
    else:
        print("Loading remote sentiment...")
        ov_llm = ChatOpenAI(
            model=sentiment_model_id,
            temperature=0,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2,
            api_key="provided_by_user_when_needed",  
            base_url=endpoint,
        )
    return ov_llm

def getQwenTemplateRemote(text: str):
    messages = [
    (
        "system",
        "You are trained to analyze and detect the sentiment of the given text. If you are unsure of an answer, you can say \"not sure\" and recommend the user review manually",
    ),
    ("user", f"Analyze the following text and determine if the sentiment is: Positive, Upset, Negative, or Neutral. {text}"
    )]
    return messages

def getQwenTemplateLocal(text: str):
    template = f"""<|im_start|>system\nYou are trained to analyze and detect the sentiment of the given text. Assume the grammar format based on the context of the text. If you are unsure of an answer, you can recommend the user review manually<|im_end|>\n<|im_start|>user\nIs the sentiment for the provided text '{text}' Positive, Negative, or Neutral?<|im_end|>\n<|im_start|>assistant\n"""

    return template
