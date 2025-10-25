from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# load dataset and format in form
#['problem', 'level', 'type', 'solution']
ds = load_dataset("qwedsacf/competition_math")
train_data = ds["train"]

# --- Model ---
model_name = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,            
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
    revision="main",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

# conf
lora_config = LoraConfig(
    r=16,                   # higher rank, more trainable params
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# tokenization
def tokenize_fn(batch):
    texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(batch["problem"], batch["solution"])
    ]
        
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=1024, 
    )

tokenized_ds = ds["train"].map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# training loop
training_args = TrainingArguments(
    output_dir="./lora-llama3.1-8b-competition-math",
    per_device_train_batch_size=2,     # you can increase this (try 2–4)
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    bf16=True,                         # if your GPU supports it
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

# training  
trainer = Trainer(
    model=model,
    train_dataset=tokenized_ds,
    args=training_args,
    data_collator=collator,
)

trainer.train()

# --- Save the LoRA adapter ---
model.save_pretrained("./lora-llama3.1-8b-competition-math-adapter")
