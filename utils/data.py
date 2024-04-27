from datasets import load_dataset
from .prompts import system_prompts, alpaca_prompt, vicuna_prompt


class Dataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._samantha_data = None

    @property
    def samantha_data(self):
        # Samantha dataset https://huggingface.co/datasets/cognitivecomputations/samantha-data
        # in Alpaca format
        if self._samantha_data:
            return self._samantha_data

        EOS_TOKEN = self.tokenizer.eos_token  # Must add EOS_TOKEN

        def formatting_prompts_func(examples):
            conversations = examples['conversations']
            system = system_prompts['samantha']
            texts = []
            for conv in conversations:
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                text = alpaca_prompt.format(system, conv['human'], conv['gpt']) + EOS_TOKEN
                texts.append(text)
            return {"text": texts, }

        dataset = load_dataset('wangqi777/samantha-data', 'en', split='train')
        self._samantha_data = dataset.map(formatting_prompts_func, batched=True)
        return self._samantha_data

    def __getitem__(self, dataset_id):
        return getattr(self, dataset_id)