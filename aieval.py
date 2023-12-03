import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM
import random
import json
import re
from tqdm import tqdm


class Chat:
    story_input = "In the following exercise, the student is given a beginning of a story. " \
               "The student needs to complete it into a full story. The exercise tests " \
               "the student´s language abilities and creativity. Be aware that student could write rubbish instead of actual story." \
               "The symbol *** marks " \
               "the separator between the prescribed beginning and the student’s completion:"

    feedback_replic = "This time, please provide your general assessment (not grades) about the part written by the " \
                    "student (the one after the *** symbol). Is it gramatically correct? Is " \
                    "it consistent with the beginning of the story? Pay special attention to " \
                    "whether the student manages to complete the sentence which is split in the " \
                    "middle by the separator ***"

    grading_replic = "Now, grade the student’s completion in terms of grammar, creativity, consistency " \
                    "with the story’s beginning and whether the plot makes sense. Grades must be in the " \
                    "format X/10, where X is your grade. Please be consistent with previous grades. Your answer should be in the following format: " \
                    "\nGrammar: X/10\nCreativity: Y/10\nConsistency: Z/10"

    initial_prompt = "You are a sentient, superintelligent artificial general intelligence, here to teach and assist me."

    def __init__(self, model, tokenizer, examples) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.start_token = "<|im_start|>"
        self.end_token = "<|im_end|>"
        self.super_prompt = self.get_super_prompt(examples).strip()
        '''
        out = self.model.generate(
            torch.tensor([self.tokenizer.encode(self.super_prompt)], device=device),
            max_new_tokens=750,
            repetition_penalty=1.1,
            do_sample=True,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        #    use_cache=True # it actually returns error with Mistral
        )
        # print(out)
        #print(type(out))
        self.super_prompt_keys = out.past_key_values
        '''

    def get_super_prompt(self, examples):
        random.seed(0)
        random.shuffle(examples)
        super_prompt = self.start_token + "system\n" + self.initial_prompt + self.end_token + "\n"
        for el in examples:
            super_prompt = super_prompt + self.start_token + "user\n" + self.story_input
            super_prompt = super_prompt + "\n\n" + el['prompt'] + '*** ' + el['story'].strip() + "\n\n"
            super_prompt = super_prompt + self.grading_replic + self.end_token + "\n"
            super_prompt = super_prompt + self.start_token + "assistant\n" + el['grade'].strip() + self.end_token + "\n"
        return super_prompt

    def generate(self, prompt):
        prefix = torch.tensor([self.tokenizer.encode(prompt)], device=device)
        generated = self.model.generate(
            prefix,
            max_new_tokens=750,
            repetition_penalty=1.1,
            do_sample=True,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id,
        #    past_key_values=self.past_key_values
        )

        all_chat = tokenizer.decode(generated[0])
        last_replic = tokenizer.decode(generated[0, prefix.shape[-1]:])
        return all_chat, last_replic

    def evaluate(self, story_prompt, story):
        prompt = self.super_prompt.strip() + "\n" + self.start_token + "user\n"
        prompt = prompt + self.story_input + "\n\n" + story_prompt + "*** " + story.strip()
        prompt = prompt + "\n\n" + self.feedback_replic + self.end_token + "\n"

        prompt = prompt + self.start_token + "assistant\n"
        replics = []
        all_chat, last_replic = self.generate(prompt)
        replics.append(last_replic)
        all_chat = all_chat + "\n" + self.start_token + "user\n"
        all_chat = all_chat + self.grading_replic + self.end_token + "\n" + self.start_token + "assistant\n"
        all_chat, last_replic = self.generate(all_chat)
        replics.append(last_replic)
        marks = last_replic
        grammar = None
        creativity = None
        consistency = None
        try:
            grammar = int(re.search(r"Grammar[:]?\s-?\s?(\d+)\/10", marks).group(1))
            creativity = int(re.search(r"Creativity[:]?\s-?\s?(\d+)\/10", marks).group(1))
            consistency = int(re.search(r"Consistency[:]?\s-?\s?(\d+)\/10", marks).group(1))
        except Exception as exp:
            print(marks, exp)


        return {
            "story_prompt": story_prompt,
            "story": story,
            "replics": replics,
            "grammar": grammar,
            "creativity": creativity,
            "consistency": consistency,
        }


def get_examples(text):
    parts = text.split("---")
    special = {"story", "grade", "prompt"}
    examples = []
    current = {}
    key = ""
    for el in parts:
        el = el.strip()
        if len(el) == 0:
            continue
        if el in special:
            key = el
        else:
            current[key] = el
            if len(current) == len(special):
                examples.append(current)
                current = {}
                key = ""
    return examples

with open('evaluation_examples.txt') as file:
    examples = get_examples(file.read())

gpt2xl_prompts = []
with open('comparison_artifacts/gpt2xl-checkpoint.json') as file:
    gpt2xl_prompts = json.load(file)
mycheckpoint = []
with open('comparison_artifacts/my-checkpoint.json') as file:
    mycheckpoint = json.load(file)


device = torch.device('cuda:0')

tokenizer = LlamaTokenizer.from_pretrained('teknium/OpenHermes-2.5-Mistral-7B', trust_remote_code=True)
model = MistralForCausalLM.from_pretrained(
    "teknium/OpenHermes-2.5-Mistral-7B",
    torch_dtype=torch.float16,
    device_map="auto",
    use_flash_attention_2=True,
).to(device)

chat = Chat(model, tokenizer, examples)

def evaluate_examples(examples):
    results = []
    for el in tqdm(examples, desc='all examples'):
        for t in tqdm(el['group'], desc='group'):
            res = chat.evaluate(el['prompt'], t)
            if res['grammar'] is None: # retry once just in case
                res = chat.evaluate(el['prompt'], t)
            results.append(res)
    grammar_grades = [el['grammar'] for el in results if el['grammar'] is not None]
    creativity_grades = [el['creativity'] for el in results if el['creativity'] is not None]
    consistency_grades = [el['consistency'] for el in results if el['consistency'] is not None]
    grammar = sum(grammar_grades) / (len(grammar_grades) + 1e-9)
    creativity = sum(creativity_grades) / (len(creativity_grades) + 1e-9)
    consistency = sum(consistency_grades) / (len(consistency_grades) + 1e-9)
    fails = 0
    for el in results:
        fails += el['grammar'] is None or el['creativity'] is None or el['consistency'] is None
    return results, grammar, creativity, consistency, fails


with open('comparison_artifacts/mycheckpoint-grades.json', 'w') as file:
    results, grammar, creativity, consistency, fails = evaluate_examples(mycheckpoint)
    json.dump(results, fp=file, indent=4)
    print(f'SimplerDimpler Grammar: {grammar:.2f}/10 Creativity: {creativity:.2f}/10 Consistency: {consistency:.2f}/10')
    print('FAILS', fails)

with open('comparison_artifacts/gpt2xl-grades.json', 'w') as file:
    results, grammar, creativity, consistency, fails = evaluate_examples(gpt2xl_prompts)
    json.dump(results, fp=file, indent=4)
    print(f'GPT2-XL Grammar: {grammar:.2f}/10 Creativity: {creativity:.2f}/10 Consistency: {consistency:.2f}/10')
    print('FAILS', fails)
