import torch
import json
import argparse

from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import TrainingArguments
import numpy as np
import evaluate

from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from joblib import Memory
from loguru import logger
from typing import List, Union

from datasets import Dataset

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

    def tokenize_text(examples):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    # ds = ds.map(tokenize_instruction, batched=True)
    # ds = ds.map(tokenize_input, batched=True)
    # ds = ds.map(tokenize_output, batched=True)
    ds = ds.map(tokenize_text, batched=True, remove_columns=['text'])
    block_size = 128

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


def get_device_map(model_name, device, do_int8):
    #return "auto"
    if device == "a100-40g":
        return "auto"

    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    d = {}
    for i in range(0, 4):
        d[i] = "16GiB"

    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16,
        no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer"]
    )
    del model
    print(device_map)
    return device_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/llama/hf/")
    parser.add_argument("--variant", type=str, default="7b", choices=["7b", "13b", "33b", "65b"])
    parser.add_argument(
        "--device", type=str, default="a5000-18g"
        # choices=["a100-40g", "v100-32g", "a5000-24g", "a5000-20g"], default="a5000-24g"
    )
    args = parser.parse_args()

    model_id = f"{args.model_path}/llama-{args.variant}"

    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/tokenizer/", model_max_length=512, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map=get_device_map(model_id, args.device, False),
                                                 low_cpu_mem_usage=True)
    # device_map=get_device_map(model_id, args.device, False),
    # torch_dtype=torch.int8 if args.do_int8 else torch.float16,
    # torch_dtype=torch.float16,

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples, field):
        return tokenizer(examples[field], padding="max_length", truncation=True)


    def tokenize_instruction(examples):
        return tokenize_function(examples, "instruction")


    def tokenize_input(examples):
        return tokenize_function(examples, "input")


    def tokenize_output(examples):
        return tokenize_function(examples, "output")


    ds = get_dataset()
    training_args = TrainingArguments(output_dir="trainer",
                                      per_device_train_batch_size=128,
                                      learning_rate=2e-5,
                                      num_train_epochs=3,
                                      weight_decay=1,
                                      evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"]
    )

    trainer.train()
