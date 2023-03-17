import torch
import json
import argparse

from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import TrainingArguments
import numpy as np
import evaluate
import transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from joblib import Memory
from loguru import logger
from typing import List, Union
from typing import Tuple
from datasets import Dataset
from accelerate import Accelerator

memory = Memory("./cache", verbose=0)



def gen():
    alpaca = json.load(open('alpaca_data.json', 'r'))
    for i in alpaca:
        yield i


@memory.cache
def get_dataset():
    ds = Dataset.from_generator(gen)
    df = ds.to_pandas()
    df['text'] = df['instruction'] + df['input'] + df['output']
    del df['instruction']
    del df['input']
    del df['output']
    ds = ds.from_pandas(df)

    with open('alpaca_data.txt', 'w') as f:
        for i in gen():
            text = i['instruction'] + " " + i['input'] + " " + i['output']
            text = text.replace('\n', ' ')
            f.write(text + "\n")

    def tokenize_text(examples):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    # ds = ds.map(tokenize_instruction, batched=True)
    # ds = ds.map(tokenize_input, batched=True)
    # ds = ds.map(tokenize_output, batched=True)
    ds = ds.map(tokenize_text, batched=True, remove_columns=['text'])

    block_size = 64

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    ds = ds.map(group_texts, batched=True)
    ds = ds.train_test_split(test_size=0.1)  # to be investigated but a stratified split is probably better
    return ds



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/llama/hf/")
    parser.add_argument("--variant", type=str, default="7b", choices=["7b", "13b", "33b", "65b"])

    args = parser.parse_args()

    model_id = f"{args.model_path}/llama-{args.variant}"
    print("model_id", model_id)
    tokenizer = transformers.LLaMATokenizer.from_pretrained(f"{args.model_path}/tokenizer/",
                                                            model_max_length=512,
                                                            truncation=True)
    model = transformers.LLaMAForCausalLM.from_pretrained(model_id, max_sequence_length=512, low_cpu_mem_usage=True)
    print("Model loaded")
    #model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    #tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/tokenizer/", model_max_length=512, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # device_map=get_device_map(model_id, args.device, False),
    # torch_dtype=torch.int8 if args.do_int8 else torch.float16,
    # torch_dtype=torch.float16,

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # def tokenize_function(examples, field):
    #     return tokenizer(examples[field], padding="max_length", truncation=True)
    #
    #
    # def tokenize_instruction(examples):
    #     return tokenize_function(examples, "instruction")
    #
    #
    # def tokenize_input(examples):
    #     return tokenize_function(examples, "input")
    #
    #
    # def tokenize_output(examples):
    #     return tokenize_function(examples, "output")


    ds = get_dataset()
    training_args = TrainingArguments(output_dir="trainer",
                                      per_device_train_batch_size=128,
                                      learning_rate=2e-5,
                                      num_train_epochs=3,
                                      weight_decay=1,
                                      evaluation_strategy="epoch")

    training_args = TrainingArguments(bf16=True,
                                      output_dir="trainer",
                                      num_train_epochs=3,
                                      per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1,
                                      gradient_accumulation_steps=8,
                                      evaluation_strategy="no",
                                      save_strategy="steps",
                                      save_steps=2000,
                                      save_total_limit=1,
                                      learning_rate=2e-5,
                                      weight_decay=0.,
                                      warmup_ratio=0.03,
                                      lr_scheduler_type="cosine",
                                      logging_steps=1,
                                      fsdp="full_shard auto_wrap",
                                      fsdp_transformer_layer_cls_to_wrap="LLaMADecoderLayer",
                                      tf32=True
                                      )

    print(model.config)
    #print(model)
    #print(training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"].shuffle(seed=42).select(range(10)),
        eval_dataset=ds["test"].shuffle(seed=42).select(range(10))
    )

    trainer.train()
