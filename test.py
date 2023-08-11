# Import the GPT-2 model and tokenizer from the transformers library
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 tokenizer from the pre-trained "gpt2" model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Load the GPT-2 language model with a language modeling head from the pre-trained "gpt2" model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define a function to generate text based on a given prompt
def generate_text(prompt):
    # Use the GPT-2 tokenizer to convert the text prompt into input IDs (numerical tokens)
    # return_tensors='pt' specifies that the result should be a PyTorch tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # Use the GPT-2 model to generate text from the input IDs, with a max length of 100 tokens
    # and a temperature of 0.7 to control randomness (lower value = more deterministic)
    output = model.generate(input_ids, max_length=100, temperature=0.7)
    # Decode the numerical tokens in the generated output back into human-readable text
    # skip_special_tokens=True removes any special tokens like padding or end-of-sequence tokens
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Return the generated text
    return text
