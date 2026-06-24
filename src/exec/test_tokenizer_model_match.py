from transformers import AutoTokenizer, MistralForCausalLM

model = MistralForCausalLM.from_pretrained("meta-mistral/Mistral-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-mistral/Mistral-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")