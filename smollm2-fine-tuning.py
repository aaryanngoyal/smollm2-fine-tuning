import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq)

from peft import LoraConfig, TaskType, get_peft_model


def main():
    
    # <---------model loading------------->
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name) # init tokenizer

    # initializing padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        else:
            tokenizer.add_special_token({"pad_token": '[PAD]'})

    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:1000]") # load dataset

    # <---------preprocessing dataset------------>
    def preprocess(ex):

        user = ex["instruction"] + (f"\n\n{ex['input']}" if ex['input'] else '')
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False)
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        a_ids = tokenizer(ex["output"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]

        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids

        max_len = 2048
        if len(ids) > max_len:
            ids = ids[:max_len]
            labels = labels[:max_len]

        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
        }

    dataset = dataset.map(
        preprocess, 
        remove_columns=dataset.column_names,
    )

    dataset = dataset.train_test_split(test_size=0.1, seed=0)

    # <---------tuning hyperparameters for trainer------------>
    training_args = TrainingArguments(
        output_dir="smollm2-sft",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_steps=250,
        logging_steps=20,
        save_steps=50,
        eval_steps=50,
        learning_rate=1e-4,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
    )   

    # initializing LoRA instance
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,  
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    # initializing model
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.bfloat16,
                                                dtype="auto")
    model.enable_input_require_grads()  
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # initializing trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                              padding=True,
                                              pad_to_multiple_of=8,
                                              label_pad_token_id=-100),
    )

    trainer.train()

if __name__ == "__main__":
    main()  