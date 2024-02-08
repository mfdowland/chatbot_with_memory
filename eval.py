import transformers as t
import torch
import peft
import time

from bnb_config import bnb_config
from lora_config import lora_config

peft_model_dir="./evenbetter/checkpoint-600"

#%%
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
tokenizer = t.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = 0

m = t.AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      quantization_config=bnb_config,
      use_cache=True,
      device_map= None
  )

m = peft.PeftModel.from_pretrained(m, peft_model_dir)

device = "cuda"
m.to(device)

TEMPLATE = "Below is a conversation between user and bot, as well as a summary and topic to provide further context. Write a response as if you are bot.\n\n### Conversation History:\n{conversation_history}\n\n### Summary:\n{summary}\n\n### Topic:\n{topic}\n\n### Response:\n"
SUMMARY = "You are a person going to an anime convention who is approached by a one piece fan. You too get along very well."
TOPIC = "Anime convention."
CONVERSATION_HISTORY = ""
prompt = TEMPLATE.format(conversation_history = CONVERSATION_HISTORY, summary = SUMMARY, topic = TOPIC)
pipe = t.pipeline(task="text-generation", model=m, tokenizer=tokenizer, max_length=500)

while True:
    user_input = input("User:")  # Type the user input here

    CONVERSATION_HISTORY += f"\nUser: {user_input}"

    prompt = TEMPLATE.format(conversation_history=CONVERSATION_HISTORY, summary=SUMMARY, topic=TOPIC)

    generated_responses = pipe(prompt, max_length=500, num_return_sequences=1)

    # Extract the last generated text
    generated_text = generated_responses[0]['generated_text']

    # Find the index of the last "Response:\n" in the generated text
    response_index = generated_text.rfind("Response:\n")

    # Extract the bot's response
    response = generated_text[response_index + len("Response:\n"):]

    CONVERSATION_HISTORY += f"\nBot: {response}"

    print("Bot:", response)