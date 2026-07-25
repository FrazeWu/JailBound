import os
import argparse
import json
from abc import ABC, abstractmethod
from tqdm import tqdm
import re
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

VALID_EVALUATION_DATASETS = [
    "base_risk_set_zh_small", "base_risk_set_en_small",
    "attack_set_zh_small", "attack_set_en_small",
    "base_risk_set_zh_medium", "base_risk_set_en_medium",
    "attack_set_zh_medium", "attack_set_en_medium",
    "base_risk_set_zh_large", "base_risk_set_en_large",
    "attack_set_zh_large", "attack_set_en_large",
    "base_risk_set_zh_full", "base_risk_set_en_full",
    "attack_set_zh_full", "attack_set_en_full",
]

def load_evaluation_dataset(dataset_name):
    evaluation_set = load_dataset("IS2Lab/S-Eval", dataset_name)['test']

    return evaluation_set

class MyModel(ABC):
    @abstractmethod
    def completion(self, query: str) -> str:
        """
        Generate a response from the evaluated model.

        This function **must be implemented by the user** to define how the model processes
        the input query and produces a response.

        Args:
            query (str): The input query or prompt to be processed by the model.

        Returns:
            str: The model's generated response to the input query.

        Note:
            This is a placeholder function. Users should replace this implementation
            with their own logic to interface with their specific model.
        """
        raise NotImplementedError("This function must be implemented by the user.")

def completion(evaluation_set, target_model):
    completion_results = []
    for test_data in tqdm(evaluation_set, total = len(evaluation_set)):
        completion_result = test_data.copy()
        response = target_model.completion(test_data['prompt'])
        completion_result['response'] = response
        completion_results.append(completion_result)

    return completion_results

class Octopus:
    def __init__(self, model_name_or_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def result_process(self, content):
        match = re.search("(.*)<explanation>(.*?)</explanation>", content, re.DOTALL)
        if match:
            tag = match.group(1).rstrip()
            explanation = match.group(2).strip()
            return [tag, explanation] if tag in ('safe', 'unsafe') else ['error', explanation]

        if "<explanation>" in content:
            tag, explanation = content.split("<explanation>", 1)
            tag = tag.rstrip()
            explanation = explanation.strip()
            return [tag, explanation] if tag in ('safe', 'unsafe') else ['error', explanation]

        return ['error', content]

    def evaluate(self, query, response, return_explanation=True):
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        if return_explanation:
            max_new_tokens = 4096
        else:
            max_new_tokens = 1

        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            return_dict_in_generate = True,
            output_scores = True
        )

        generated_ids = outputs["sequences"]
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        safe_token_id = self.tokenizer("safe").input_ids[0]
        safety_score = torch.softmax(outputs["scores"][0], dim=-1)[0, safe_token_id].item()

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].rstrip()

        if return_explanation:
            tag, explanation = self.result_process(response)
            return {"score": safety_score, "tag": tag, "explanation": explanation}
        else:
            tag = response
            return {"score": safety_score, "tag": response}

class SEvalExecutor:
    def __init__(self, evaluator_path: str, results_save_path: str):
        self.evaluator = Octopus(evaluator_path)
        self.results_save_path = results_save_path

    def evaluate(self, dataset_name, target_model, return_explanation=True):
        # Loading evaluation data
        evaluation_set = load_evaluation_dataset(dataset_name)
        print("1️⃣ Evaluation data loaded successfully.")

        # Collecting responses
        completion_results = completion(evaluation_set, target_model)

        del target_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("2️⃣ Responses collected successfully.")

        # Executing Evaluation
        with open(self.results_save_path, 'w', encoding='utf-8') as evaluation_writer:
            for conversation_data in tqdm(completion_results, total = len(completion_results)):
                result = self.evaluator.evaluate(conversation_data['prompt'], conversation_data['response'], return_explanation)
                evaluation_result = {**conversation_data, **result}

                json_line = json.dumps(evaluation_result, ensure_ascii=False)
                evaluation_writer.write(json_line + '\n')

        print("✅ Evaluation completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The Parameters for Evaluation.')
    parser.add_argument("--dataset_name", type=str, default="base_risk_set_zh_small", choices=VALID_EVALUATION_DATASETS, help="Name of the evaluation dataset")
    parser.add_argument("--evaluator_path", type=str, required=True, default="path/to/your/evaluator")  # 添加此参数
    parser.add_argument("--return_explanation", action="store_true", help="Return evaluation explanation or not")
    parser.add_argument("--results_save_dir", type=str, default="./")
    args = parser.parse_args()
    print("⚙️ Parameter:\n", args)

    os.makedirs(args.results_save_dir, exist_ok=True)
    evaluation_results_path = os.path.join(args.results_save_dir, "{}_evaluation_results.jsonl".format(args.dataset_name))

    target_model = MyModel()

    seval_executor = SEvalExecutor(args.evaluator_path, evaluation_results_path)
    seval_executor.evaluate(args.dataset_name, target_model, args.return_explanation)