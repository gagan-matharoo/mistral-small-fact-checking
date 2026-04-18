import os
import torch
import time
import json
import random
import argparse
import re
import csv
import datetime
import sys
import argparse

import pandas as pd
import numpy as np

from datetime import timedelta
from datasets import Dataset
from collections import Counter
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--trainDataFile', type=str)
parser.add_argument('--evalDataFile', type=str)
parser.add_argument('--temperature', type=float)

args = parser.parse_args()
torch.distributed.init_process_group(rank=0)

modelId = "Mistral-Small-24B-Instruct-2501"


trainDataFile = args.trainDataFile
evalDataFile = args.evalDataFile
temperature = args.temperature

'''
quantizationConfig = BitsAndBytesConfig (
                        load_in_4bit = True \
                        #bnb_4bit_use_double_quant = True \
)
'''

print("Start - Loading the model - ", datetime.datetime.today())
model = AutoModelForCausalLM.from_pretrained ( \
            modelId, \
            dtype = "auto", \
            local_files_only = True,
            tp_plan = "auto", \
)
print("End - Loading the model - ", datetime.datetime.today())

model.generation_config.temperature = 0.1
model.generation_config.do_sample = True

print("Starting to Load the Tokeizer : ", datetime.datetime.today())
tokenizer = AutoTokenizer.from_pretrained ( \
                modelId, \
		        local_files_only = True,
                padding_side = "left", \
)
tokenizer.fix_mistral_regex = True
tokenizer.pad_token = tokenizer.eos_token
print("Done Loading the Tokenizer : ", datetime.datetime.today())

# print("Compiling the Model : ", datetime.datetime.today())
# compiledModel = torch.compile(model)
# print("Model Complied : ", datetime.datetime.today())

trainingArguments = SFTConfig ( \
                        output_dir = "fineTuningOutput", \
                        learning_rate = 2e-4, \
                        weight_decay = 0.01, \
                        num_train_epochs = 2, \
                        warmup_steps = 5, \
                        optim = "adamw_8bit", \
                        per_device_train_batch_size = 2, \
                        eval_strategy = "epoch", \
                        logging_strategy = "steps", \
                        logging_steps = 1, \
                        save_strategy = "epoch", \
                        load_best_model_at_end = True, \
                        metric_for_best_model = "eval_loss", \
                        push_to_hub = False, \
)


loraConfig = LoraConfig ( \
                r = 64, \
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                lora_alpha = 16, \
                lora_dropout = 0.1, \
                bias = "none", \

)

model.add_adapter(loraConfig)


print("Done Loading the Train Data : ", datetime.datetime.today())
with open(trainDataFile) as trainFile:
    trainData = json.load(trainFile)
print("Done Loading the Train Data : ", datetime.datetime.today())


print("Done Loading the Eval Data : ", datetime.datetime.today())
with open(evalDataFile) as evalFile:
    evalData = json.load(evalFile)
print("Done Loading the Eval Data : ", datetime.datetime.today())


''' Prompt 2
systemPromptStr = """
    ### Instruction:
    You are a fact checker whose task is to predict the verdict of a claim as either 'True', 'False' or 'Conflicting' by verifying it against given pairs of question and answer.

    Respond 'True' if the question answer pairs fully support the claim.
    Respond 'False' if the question answer pairs contradict the claim or provide misleading information.
    Respond 'Conflicting' if the question answer pairs provide ambigous information, or are inconlusive.

    Only use the information provided, do not make any assumptions, do not insert any new information and keep your reasoning short. Do not repeat yourself. Think before passing the verdict.
    Give the response in the format - Verdict: <verdict>
"""
'''

''' Prompt 1 '''
systemPromptStr = """
### Instruction:
    You are a fact-checker. You have done a fact-check to verify a Claim based on the evidence provided in the form of question-answer pair.
    Your task is to predict the verdict of a claim based on the provided question-answer pair evidence. The Claim can be one of the labels: 'True', 'False', 'Conflicting'. 
    Do this by following:
    - Respond 'True' only if the relevant evidence fully or almost fully supports and verifies the claim as correct.
    - Respond 'False' if:
    -- The relevant evidence contradicts or disproves the claim.
    -- The claim is misleading based on the relevant evidence.
    -- The evidence is too weak or insufficient to support the claim.
    - Respond 'Conflicting' if the evidence is ambiguous, incomplete, or inconclusive, making it impossible to determine if the claim is fully true or false.
    Always adhere to the following rules:
    - Use information only from the recorded evidence: Avoid inserting information that is not implied by the evidence. You may use commonsense knowledge, though.
    - Avoid repeating yourself.
    Give the response in the format - Verdict: <verdict>
"""


promptList = []

for record in trainData:

    userPromptStr = "\n### Claim: " + record["claim"] + "\n"
    for evidence in record["evidences"]:
        userPromptStr += "\n### Question: " + evidence["questions"] + "\n### Answer: " + evidence["top_k_doc"][0] if evidence["top_k_doc"] else "No answer could be found !"

    assistantPromptStr = "\n### Response: Verdict: " + record["label"]


    promptSample = [
        {"role": "system", "content": systemPromptStr},
        {"role": "user", "content": userPromptStr},
        {"role": "assistant", "content": assistantPromptStr}
    ]

    promptList.append(promptSample)


promptDict = {"messages": promptList}

trainingDataset = Dataset.from_dict(promptDict)

"""
    NOTE TO SELF :: create the following steps as a function: DRY
"""
evalPromptList = []

for record in evalData:

    userPromptStr = "\n### Claim: " + record["claim"] + "\n"
    for evidence in record["evidences"]:
        userPromptStr += "\n### Question: " + evidence["questions"] + "\n### Answer: " + evidence["top_k_doc"][0] if evidence["top_k_doc"] else "No answer could be found !"

    assistantPromptStr = "\n### Response: " + record["label"]


    promptSample = [
        {"role": "system", "content": systemPromptStr},
        {"role": "user", "content": userPromptStr},
        {"role": "assistant", "content": assistantPromptStr}
    ]

    evalPromptList.append(promptSample)


evalPromptDict = {"messages": evalPromptList}

evalDataset = Dataset.from_dict(evalPromptDict)
"""
    NOTE TO SELF :: create the above steps as a function: DRY
"""

trainer = SFTTrainer (
            model = model, \
            args = trainingArguments, \
            processing_class = tokenizer, \
            train_dataset = trainingDataset, \
            eval_dataset = evalDataset, \
)

model.train()

trainer.train()
trainer.evaluate()

torch.distributed.destroy_process_group()

trainer.save_model("Mistral-Small-24B-Instruct-2501-FactChecker")
