from peft import LoraConfig
from hyper_params import LORA_R, LORA_ALPHA, LORA_DROPOUT

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
