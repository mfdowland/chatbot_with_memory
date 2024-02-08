import transformers as t
import datasets
import torch

TEMPLATE = "Below is a conversation between user and bot, as well as a summary and topic to provide further context. Write a response as if you are bot.\n\n### Conversation History:\n{conversation_history}\n\n### Summary:\n{summary}\n\n### Topic:\n{topic}\n\n### Response:\n"
class TrainConvoData(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        self.ds = datasets.load_dataset("knkarthick/dialogsum")
        self.ds = self.ds["train"]
        self.ds = self.ds.map(self.prompt, remove_columns=["dialogue", "summary", "topic"], load_from_cache_file=False, num_proc=8) #need to remove the id column?
        self.ds = self.ds.map(self.tokenize, remove_columns=["prompt"], load_from_cache_file=False, num_proc=8)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def prompt(self, elm):
        #change dialogue to be #Person1#: etc without last line
        conversation = elm["dialogue"]

        #need conversation history and last one separate
        last_line_index = conversation.rfind("#Person2#")
        convo_history = conversation[:last_line_index]
        convo_history = convo_history.replace('#Person1#', "\n#user#").replace('#Person2#', "\n#bot#")
        reply = conversation[last_line_index + 11:] #reply without #Person2#:
        prompt = TEMPLATE.format(conversation_history = convo_history, summary = elm["summary"], topic = elm["topic"])
        prompt = prompt + reply
        return {"prompt": prompt}

    def tokenize(self, elm):
        res = self.tokenizer(elm["prompt"])
        res["input_ids"].append(self.tokenizer.eos_token_id)
        res["attention_mask"].append(1)
        res["labels"] = res["input_ids"].copy()
        return res
    
    def max_seq_len(self):
        return max([len(elm["input_ids"]) for elm in self.ds])