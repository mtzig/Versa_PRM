import math
import statistics
from typing import Optional
from peft import PeftModel
import torch,os
from .prm_interface import PRM, StepScore
from transformers import BitsAndBytesConfig
from torch.types import Device
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import json
import os
def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

from tqdm import tqdm

class test_prm_dual(PRM):
    def __init__(
        self,
        aggregation: str = "full",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
        model_id: str = "Qwen/Qwen2.5-Math-7B-PRM"
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.quantization_config = quantization_config
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if self.quantization_config:
            raise NotImplementedError
        else:
            self.model = AutoModel.from_pretrained(
            self.model_id, 
            device_map=self.device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            ).eval()
        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> list[float]:
        """
        Computes scores for each reasoning step in the single_beam.

        Args:
            single_beam (str): A single reasoning beam, consisting of Question + Solution.

        Returns:
            list[float]: The scores for each step in the Solution.
        """
        ###

        input_ids = self.tokenizer.encode(
            single_beam, 
            return_tensors="pt", 
        ).to(self.model.device)

        outputs = self.model(input_ids=input_ids)
        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)

        # print(token_masks)

        step_reward = make_step_rewards(outputs[0], token_masks)
        # print(step_reward)
        # input()
        step_probs = step_reward[0]

        ###
        if self.aggregation == "min":
            return min(step_probs)
        elif self.aggregation == "max":
            return max(step_probs)
        elif self.aggregation == "mean":
            return statistics.mean(step_probs)
        elif self.aggregation == "prod":
            return math.prod(step_probs)
        elif self.aggregation == "last":
            return step_probs[-1]
        elif self.aggregation == "full":
            return step_probs
        else:
            raise NotImplementedError


    def __call__(self, steps: list[str]) -> list[StepScore]:
        """
        Computes scores for a list of reasoning beams.

        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of StepScore objects, each containing step and score.
        """
        result = []

        for beam in steps:
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result

if __name__ == "__main__":
    prm = test_prm_dual(
                aggregation="full", 
            )
    
    json_file_path = "/home/ec2-user/strawberry/prmMixedDomain/prmMixedDomain.json"
    data = read_json_file(json_file_path)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-PRM800K", trust_remote_code=True)

    for each_data in tqdm(data):
        for cot in each_data["chain_of_thoughts"]:
            steps = cot["steps"]
            steps = [step.replace("<extra_0>", "") for step in steps]
            question = each_data["question"].replace("<extra_0>", "")
            # updated_steps = []
            # for index, step in enumerate(steps):
            #     indexed_step = f"{step} \n\n\n\n"
            #     updated_steps.append(indexed_step)
            # steps = updated_steps
            # steps_all = f"{question} \n\n" + "".join(steps)
            messages = [
    {"role": "system", "content": "Please reason step by step."},
    {"role": "user", "content": question},
    {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},]
            steps_all = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
            )
            rewards = prm([steps_all])
            cot["prm_reward"] = rewards[0].score
            print(cot["prm_reward"])
            print(len(cot["prm_reward"]))
            print(len(steps))
            input()
    

