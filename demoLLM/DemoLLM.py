from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2",output_hidden_states=True)

text = "Who is Donald Trump?he is the president of the "

#tokenize the input string
input = tokenizer.encode(text, return_tensors="pt")

#run the model
output = model.generate(input, max_length=50, do_sample=False)

#print the output
print('\n', tokenizer.decode(output[0], skip_special_tokens=True))


####