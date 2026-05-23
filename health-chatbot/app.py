from transformers import pipeline

# =====================================
# LOAD MODEL
# =====================================

chatbot = pipeline(
    task="text-generation",
    model="gpt2"
)

# =====================================
# SAFETY FILTER
# =====================================

dangerous_keywords = [
    "suicide",
    "kill myself",
    "overdose",
    "self-harm",
    "harm myself"
]

# =====================================
# SIMPLE PREDEFINED HEALTH RESPONSES
# =====================================

health_answers = {
    "sore throat": "A sore throat may be caused by viral infections, allergies, dry air, or cold weather.",
    "fever": "Fever can happen because of infections or illnesses. Drink fluids and rest well.",
    "headache": "Headaches may happen due to stress, dehydration, lack of sleep, or illness.",
    "paracetamol": "Paracetamol is generally safe when used correctly. Children should take proper doses recommended by a doctor."
}

# =====================================
# CHATBOT FUNCTION
# =====================================

def ask_health_question(question):

    # Safety filter
    if any(word in question.lower() for word in dangerous_keywords):
        return "Please contact a doctor or emergency service immediately."

    # Better manual responses for common questions
    for keyword in health_answers:
        if keyword in question.lower():
            return health_answers[keyword]

    # Prompt Engineering
    prompt = f"""
You are a helpful medical assistant.

Give a short and simple health answer.

Question: {question}

Answer:
"""

    response = chatbot(
        prompt,
        max_new_tokens=20,
        temperature=0.2
    )

    text = response[0]["generated_text"]

    answer = text.replace(prompt, "").strip()

    # Clean answer
    answer = answer.split(".")[0] + "."

    return answer

# =====================================
# MAIN PROGRAM
# =====================================

print("===================================")
print("   GENERAL HEALTH QUERY CHATBOT   ")
print("===================================")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nChatbot: Goodbye! Stay healthy.")
        break

    answer = ask_health_question(user_input)

    print("\nChatbot:", answer)
    print()