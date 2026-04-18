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

import pandas as pd
import numpy as np

from datasets import Dataset
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from accelerate import PartialState
from accelerate.utils import gather_object
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

torch.cuda.empty_cache()


modelId = "Mistral-Small-24B-Instruct-2501-FactChecker"
testDataFile = "data/testData.json"
evidenceCorpus = "data/evidenceCorpus.json"


print("Loading the Model : ", datetime.datetime.today())
model = AutoModelForCausalLM.from_pretrained ( \
            modelId, \
            local_files_only = True, \
            torch_dtype = "auto", \
            tp_plan = "auto" \
)
print("Done Loading the Model : ", datetime.datetime.today())


print("Starting to Load the Tokeizer : ", datetime.datetime.today())
tokenizer = AutoTokenizer.from_pretrained ( \
                modelId, \
                local_files_only = True, \
                padding_side = "left"
)

tokenizer.pad_token = "<myPadToken>"
print("Done Loading the Tokenizer : ", datetime.datetime.today())


print("Starting Loading Data : ", datetime.datetime.today())
with open(testDataFile) as testFile:
    testData = json.load(testFile)
print("Done Loading the Data : ", datetime.datetime.today())

print("Start Loading Evidence Corpus : ", datetime.datetime.today())
with open(evidenceCorpus) as corpusFile:
    evidenceData = json.load(corpusFile)
print("Done Loading Evidence Corpus : ", datetime.datetime.today())


testPromptList = []

''' Prompt 1
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
'''


''' Prompt 2 '''
systemPromptStr = """
    ### Instruction:
    You are a fact checker whose task is to predict the verdict of a claim as either 'True', 'False' or 'Conflicting' by verifying it against given pairs of question and answer.

    Respond 'True' if the question answer pairs fully support the claim.
    Respond 'False' if the question answer pairs contradict the claim or provide misleading information.
    Respond 'Conflicting' if the question answer pairs provide ambigous information, or are inconlusive.

    Only use the information provided, do not make any assumptions, do not insert any new information and keep your reasoning short. Do not repeat yourself. Think before passing the verdict.
    Give the response in the format - Verdict: <verdict>
"""


random.seed(16)
for record in testData:

    userPromptStr = "\n### Claim: " + record["claim"] + "\n"


    for evidence in record["evidences"]:
        randomIndex = random.randint(0, 308399) # total evidence present 308400
        userPromptStr += "\n### Question: " + evidence["questions"] + "\n### Answer: " + evidenceData["evidenceCorpus"][randomIndex]


    userPromptConverted = [
        {"role": "system", "content": systemPromptStr},
        {"role": "user", "content": userPromptStr}
    ]

    promptWithChatTemplate = tokenizer.apply_chat_template ( \
                                userPromptConverted, \
                                tokenize = False, \
                                add_generation_prompt = True, \
    )

    testPromptList.append(promptWithChatTemplate)


verdicts = []

for prompt in testPromptList:
    inputs = tokenizer.encode (
        prompt, \
        return_tensors = "pt", \
        padding = True, \
    ).to("cuda")

    print("*************************************INPUTS****************************************")
    print(inputs)

    model.eval()

    outputs = model.generate (
        input_ids = inputs, \
        use_cache = True, \
        max_new_tokens = 500, \
        temperature = 0.3, \
        top_p = 0.9, \
        top_k = 10, \
        pad_token_id = tokenizer.pad_token_id, \
    )

    decodedOutput = tokenizer.decode (
        outputs[0], \
        skip_special_tokens = True, \
    )

    print("******************************OUTPUT*****************************")
    print(decodedOutput)

    verdictIndex = decodedOutput.rfind("### Response: Verdict: ")
    if verdictIndex == -1:
        verdicts.append("No response")
    else:
        verdicts.append(decodedOutput[verdictIndex:])


print("******************************VERDICTS*****************************")
for verdict in verdicts:
    print(verdict)

    
torch.distributed.destroy_process_group()
