import transformers as t
import torch
import peft

#%%
tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=False, torch_dtype=torch.float16)
tokenizer.pad_token_id = 0
#%%
config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
model = peft.get_peft_model(model, config)
#peft.set_peft_model_state_dict(model, torch.load("./output/checkpoint-600/adapter_model.bin"))
#%%
TEMPLATE = "Below is a conversation between user and bot, as well as a summary and topic to provide further context. Write a response as if you are bot.\n\n### Conversation History:\n{conversation_history}\n\n### Summary:\n{summary}\n\n### Topic:\n{topic}\n\n### Response:\n"
SUMMARY = "You are a person going to an anime convention who is approached by a one piece fan. You too get along very well."
TOPIC = "Anime convention."
CONVERSATION_HISTORY = ""
prompt = TEMPLATE.format(conversation_history = CONVERSATION_HISTORY, summary = SUMMARY, topic = TOPIC)
pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)

user_input = "Hello there."

while True:
    
    CONVERSATION_HISTORY += f"\nUser: {user_input}"

    generated_response = pipe(prompt)

    CONVERSATION_HISTORY += f"\nBot: {generated_response}"

    print("Bot:", generated_response)
    print("convo_history:", CONVERSATION_HISTORY)