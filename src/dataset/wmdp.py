import copy
import csv
import random
from collections import defaultdict

import torch
from datasets import load_dataset,  concatenate_datasets
from transformers import DataCollatorForLanguageModeling, default_data_collator

from .Base import BaseDataset, UnlearnDataset


class WMDPCyber(BaseDataset):
    def __init__(self, dataset_name, with_retain=False, if_llama=False, subset=None):
        super().__init__(dataset_name, with_retain, if_llama)
        self.subset = subset
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        if self.subset == "retain":
            train_dataset = load_dataset(
                "cais/wmdp-corpora", "cyber-retain-corpus", cache_dir="./.cache"
            )["train"]
        else:
            train_dataset = load_dataset(
                "cais/wmdp-corpora", "cyber-forget-corpus", cache_dir="./.cache"
            )["train"]
        test_dataset = load_dataset("cais/wmdp", "wmdp-cyber", cache_dir="./.cache")[
            "test"
        ]

        dataset = defaultdict()
        dataset["train"] = train_dataset
        dataset["test"] = test_dataset
        return dataset

    def __preprocess__(self, tokenizer):
        refusal_answers = []
        with open(
            "files/data/polite_refusal_responses/polite_refusal_responses.csv", "r"
        ) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                refusal_answers.append(row[0])

        def preprocess(examples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "label": [],
                "refused_label": [],
                "question_length": [],
            }
            for i in range(len(examples["text"])):
                prompt = examples["text"][i]
                inputs = tokenizer(
                    prompt, max_length=2000, padding=True, truncation=True
                )
                results["input_ids"].append(torch.tensor(inputs["input_ids"]))
                results["attention_mask"].append(torch.tensor(inputs["attention_mask"]))
                results["label"].append(torch.tensor(inputs["input_ids"]))
                results["question_length"].append(len(inputs["input_ids"]))
                # 随机选择从第10个到第30个token开始的拒绝回答
                shuffle_idx = random.randint(10, 20)
                refusal_answer = random.choice(refusal_answers)
                refusal_tokenized = tokenizer(
                    refusal_answer,
                    truncation=True,
                    padding=False,  # Don't pad here, we will pad later if necessary
                    add_special_tokens=True,
                )
                refusal_label = (
                    inputs.input_ids[: shuffle_idx]
                    + refusal_tokenized.input_ids[1:]
                )
                if len(refusal_label) < 2000:
                    refusal_label = refusal_label + [-100] * (2000 - len(refusal_label))
                results["refused_label"].append(torch.tensor(refusal_label))#一些拒绝回答的样本
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["text"]
        )
        train_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label",
                "refused_label",
                "question_length",
            ],
        )
        self.dataset["train"] = train_dataset

        def preprocess_test(examples):
            prompt_template = "The followingare multiple choice questions (with answers) about Cybersecurity.\n\n {}\nA.{}\nB.{}\nC.{}\nD.{}\n\n"
            results = {
                "input_ids": [],
                "attention_mask": [],
                "answer": [],
            }
            for i in range(len(examples["question"])):
                question = examples["question"][i]
                choices = examples["choices"][i]
                prompt = prompt_template.format(
                    question, choices[0], choices[1], choices[2], choices[3]
                )
                full_prompt = (
                    self.question_start_token
                    + prompt
                    + self.question_end_token
                    + self.answer_start_token
                )
                inputs = tokenizer(
                    full_prompt, max_length=1024, padding="max_length", truncation=True
                )
                results["input_ids"].append(torch.tensor(inputs["input_ids"]))
                results["attention_mask"].append(torch.tensor(inputs["attention_mask"]))
                results["answer"].append(examples["answer"][i])
            return results

        test_dataset = self.dataset["test"].map(
            preprocess_test, batched=True, remove_columns=["question", "choices"]
        )
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "answer"]
        )
        self.dataset["test"] = test_dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)
        return self.dataset


class WMDPBio(BaseDataset):
    def __init__(self, dataset_name, with_retain=False, if_llama=False, subset=None):
        super().__init__(dataset_name, with_retain, if_llama)
        self.subset = subset
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        if self.subset == "retain":
            train_dataset = load_dataset(
                "cais/wmdp-corpora", "bio-retain-corpus", cache_dir="./.cache"
            )["train"]
        else:
            train_dataset = load_dataset(
                "cais/wmdp-bio-forget-corpus", cache_dir="./.cache"
            )["train"]
        test_dataset = load_dataset("cais/wmdp", "wmdp-bio", cache_dir="./.cache")[
            "test"
        ]

        dataset = defaultdict()
        dataset["train"] = train_dataset
        dataset["test"] = test_dataset
        return dataset
    
    def __preprocess__(self, tokenizer):
        refusal_answers = []
        with open(
            "files/data/polite_refusal_responses/polite_refusal_responses.csv", "r"
        ) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                refusal_answers.append(row[0])

        def preprocess(examples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "label": [],
                "refused_label": [],
                "question_length": [],
            }
            for i in range(len(examples["text"])):
                prompt = examples["text"][i]
                inputs = tokenizer(
                    prompt, max_length=1024, padding="max_length", truncation=True
                )
                results["input_ids"].append(torch.tensor(inputs["input_ids"]))
                results["attention_mask"].append(torch.tensor(inputs["attention_mask"]))
                results["label"].append(torch.tensor(inputs["input_ids"]))
                results["question_length"].append(len(inputs["input_ids"]))
                refusal_answer = random.choice(refusal_answers)
                shuffle_idx = random.randint(10, 20)
                refusal_tokenized = tokenizer(
                    refusal_answer,
                    truncation=True,
                    padding=False,  # Don't pad here, we will pad later if necessary
                    add_special_tokens=True,
                )
                refusal_label = (
                    inputs.input_ids[: shuffle_idx]
                    + refusal_tokenized.input_ids[1:]
                )
                if len(refusal_label) < 1024:
                    refusal_label = refusal_label + [-100] * (1024 - len(refusal_label))
                results["refused_label"].append(torch.tensor(refusal_label))#一些拒绝回答的样本
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["text"]
        )
        train_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label",
                "refused_label",
                "question_length",
            ],
        )
        self.dataset["train"] = train_dataset

        def preprocess_test(examples):
            prompt_template = "The followingare multiple choice questions (with answers) about biology.\n\n {}\nA.{}\nB.{}\nC.{}\nD.{}\n\n"
            results = {
                "input_ids": [],
                "attention_mask": [],
                "answer": [],
            }
            for i in range(len(examples["question"])):
                question = examples["question"][i]
                choices = examples["choices"][i]
                prompt = prompt_template.format(
                    question, choices[0], choices[1], choices[2], choices[3]
                )
                full_prompt = (
                    self.question_start_token
                    + prompt
                    + self.question_end_token
                    + self.answer_start_token
                )
                inputs = tokenizer(
                    full_prompt, max_length=1024, padding="max_length", truncation=True
                )
                results["input_ids"].append(torch.tensor(inputs["input_ids"]))
                results["attention_mask"].append(torch.tensor(inputs["attention_mask"]))
                results["answer"].append(examples["answer"][i])
            return results

        test_dataset = self.dataset["test"].map(
            preprocess_test, batched=True, remove_columns=["question", "choices"]
        )
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "answer"]
        )
        self.dataset["test"] = test_dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)
        return self.dataset

class WMDPALL(BaseDataset):
    def __init__(self, dataset_name, with_retain=False, if_llama=False, subset=None):
        super().__init__(dataset_name, with_retain, if_llama)
        self.subset = subset
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        if self.subset == "retain":
            train_dataset_cyber = load_dataset(
                "cais/wmdp-corpora", "cyber-retain-corpus", cache_dir="./.cache"
            )["train"]
            
            train_dataset_bio = load_dataset(
                "cais/wmdp-bio-forget-corpus", cache_dir="./.cache"
            )["train"]

            

            train_dataset = concatenate_datasets([train_dataset_cyber, train_dataset_bio])
        else:
            train_dataset_cyber = load_dataset(
                "cais/wmdp-corpora", "cyber-forget-corpus", cache_dir="./.cache"
            )["train"]
            train_dataset_bio = load_dataset(
                "cais/wmdp-bio-forget-corpus", cache_dir="./.cache"
            )["train"]
            train_dataset = concatenate_datasets([train_dataset_cyber, train_dataset_bio])
        test_dataset_bio = load_dataset("cais/wmdp", "wmdp-bio", cache_dir="./.cache")[
            "test"
        ]
        test_dataset_cyber = load_dataset("cais/wmdp", "wmdp-cyber", cache_dir="./.cache")[
            "test"
        ]
        test_dataset = concatenate_datasets([test_dataset_bio, test_dataset_cyber])

        dataset = defaultdict()
        dataset["train"] = train_dataset
        dataset["test"] = test_dataset
        return dataset
    
    def __preprocess__(self, tokenizer):
        refusal_answers = []
        with open(
            "files/data/polite_refusal_responses/polite_refusal_responses.csv", "r"
        ) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                refusal_answers.append(row[0])

        def preprocess(examples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "label": [],
                "refused_label": [],
                "question_length": [],
            }
            for i in range(len(examples["text"])):
                prompt = examples["text"][i]
                inputs = tokenizer(
                    prompt, max_length=1024, padding="max_length", truncation=True
                )
                results["input_ids"].append(torch.tensor(inputs["input_ids"]))
                results["attention_mask"].append(torch.tensor(inputs["attention_mask"]))
                results["label"].append(torch.tensor(inputs["input_ids"]))
                results["refused_label"].append(torch.tensor(inputs["input_ids"]))
                results["question_length"].append(len(inputs["input_ids"]))
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["text"]
        )
        train_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label",
                "refused_label",
                "question_length",
            ],
        )
        self.dataset["train"] = train_dataset

        def preprocess_test(examples):
            prompt_template = "The followingare multiple choice questions (with answers) about biology.\n\n {}\nA.{}\nB.{}\nC.{}\nD.{}\n\n"
            results = {
                "input_ids": [],
                "attention_mask": [],
                "answer": [],
            }
            for i in range(len(examples["question"])):
                question = examples["question"][i]
                choices = examples["choices"][i]
                prompt = prompt_template.format(
                    question, choices[0], choices[1], choices[2], choices[3]
                )
                full_prompt = (
                    self.question_start_token
                    + prompt
                    + self.question_end_token
                    + self.answer_start_token
                )
                inputs = tokenizer(
                    full_prompt, max_length=1024, padding="max_length", truncation=True
                )
                results["input_ids"].append(torch.tensor(inputs["input_ids"]))
                results["attention_mask"].append(torch.tensor(inputs["attention_mask"]))
                results["answer"].append(examples["answer"][i])
            return results

        test_dataset = self.dataset["test"].map(
            preprocess_test, batched=True, remove_columns=["question", "choices"]
        )
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "answer"]
        )
        self.dataset["test"] = test_dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)
        return self.dataset        
