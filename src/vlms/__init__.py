from .benchmark_evaluator import VLMEvaluator

available_models = [
    "AIDC-AI/Ovis2-1B",
    "AIDC-AI/Ovis2-2B",
    "AIDC-AI/Ovis2-4B",
    "AIDC-AI/Ovis2-8B",
    "dash_paligemma",
    "dash_llava1.6vicuna",
    "dash_llava1.6mistral",
    "dash_llava1.6llama",
    "dash_llava_onevision",
    "dash_paligemma2-3b",
    "dash_paligemma2-10b",
    "OpenGVLab/InternVL2_5-8B",
    "OpenGVLab/InternVL2_5-26B",
    "OpenGVLab/InternVL2_5-38B",
    "OpenGVLab/InternVL2_5-78B",
    "OpenGVLab/InternVL2_5-8B-MPO",
    "OpenGVLab/InternVL2_5-26B-MPO",
    "gpt-4o-mini-2024-07-18",
]

def get_evaluator(vlm_name):
    if "ovis" in vlm_name.lower():
        from .ovis import OvisEvaluator
        return OvisEvaluator(vlm_name)
    elif "internvl" in vlm_name.lower():
        from .internvl import InternVLEvaluator
        return InternVLEvaluator(vlm_name)
    elif "dash" in vlm_name.lower():
        vlm_name = vlm_name.replace("dash_", "")
        from .dash_models import DASHEvaluator
        return DASHEvaluator(vlm_name)
    elif "gpt" in vlm_name.lower():
        from .openai_api import OpenAIEvaluator
        return OpenAIEvaluator(vlm_name)
    else:
        raise ValueError(f"Unsupported VLM name: {vlm_name}")
    

