from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Load dataset
dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)

# Small subset for faster training
small_dataset = dataset["train"].select(range(1000))

# Model name
model_name = "distilgpt2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preprocessing function
def preprocess(example):

    text = f"User: {example['prompt']}\nBot: {example['utterance']}"

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    encoding["labels"] = encoding["input_ids"].copy()

    return encoding

# Tokenize dataset
tokenized_dataset = small_dataset.map(preprocess)

# Training settings
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=200,
    logging_steps=50,
    save_total_limit=1
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Training Complete!")
