import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Page settings
st.set_page_config(page_title="Mental Health Chatbot")

# Title
st.title("Mental Health Chatbot")
st.write("Talk with your AI mental health assistant.")

# Load fine-tuned model
model_path = "./fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Fix padding token
tokenizer.pad_token = tokenizer.eos_token


# Response function
def generate_response(user_input):

    # Predefined supportive replies
    supportive_responses = {
        "stress": "I'm sorry you're feeling stressed. Try focusing on one small step at a time. You don't have to solve everything today.",

        "future": "Thinking about the future can feel overwhelming sometimes. Remember that growth happens slowly, one day at a time.",

        "anxious": "I'm here for you. Anxiety can feel heavy, but taking deep breaths and slowing your thoughts may help.",

        "anxiety": "It's okay to feel anxious sometimes. Try taking a short pause and focus on your breathing.",

        "sad": "I'm sorry you're feeling sad. Talking to someone you trust or taking a short break can sometimes help.",

        "lonely": "Feeling lonely can be really difficult. Remember that your feelings matter and you deserve support.",

        "depressed": "I'm really sorry you're feeling this way. Please try to reach out to someone you trust and take care of yourself.",

        "tired": "You may be emotionally exhausted right now. Try to rest and be kind to yourself.",

        "overthinking": "Overthinking can become overwhelming. Try focusing only on what you can control right now.",

        "panic": "Panic can feel scary, but you're safe right now. Try taking slow deep breaths."
    }

    # Convert input to lowercase
    text = user_input.lower()

    # Keyword matching
    for keyword, reply in supportive_responses.items():
        if keyword in text:
            return reply

    # AI generation fallback
    prompt = f"""
You are a caring and supportive mental health assistant.
Give short, calm, and helpful responses.
Do not repeat words.
Keep responses under 2 sentences.

User: {user_input}
Assistant:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.5,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    # Cleanup
    response = response.replace("_comma_", ",")
    response = response.replace("\n", " ")

    # Bad output filter
    bad_words = [
        "fuck",
        "xxxx",
        "hhhh",
        "kiiiii",
        "spam",
        "dddd"
    ]

    for word in bad_words:
        if word in response.lower():
            return "I'm here for you. Please remember to take things one step at a time."

    # Empty or huge response protection
    if len(response) < 5 or len(response) > 250:
        return "I'm here for you. Things may feel difficult right now, but you're not alone."

    return response


# User input
user_input = st.text_input("You:")

# Show response
if user_input:
    response = generate_response(user_input)

    st.write(f"Bot: {response}")