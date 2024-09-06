import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# GPT-2 Model and Tokenizer initialization
model_name = "gpt2-large"  
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit framework
st.title("Langchain demo with GPT-2")

input_text = st.text_input("Enter your Query")

# If there's input from the user
if input_text:
    # Encode the input text and move it to GPU if available
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate text using GPT-2
    output = model.generate(
        input_ids, 
        max_length=120, 
        num_return_sequences=1, 
        temperature=0.7,  
        top_k=50, 
        repetition_penalty=1.3  
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Display the generated texts
    structured_text = generated_text.replace("•", "\n- ")  
    st.write(structured_text)
    
