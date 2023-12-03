from typing import Any
import torch
from datautils import *
from optimutils import *
from model import SimplerDimplerModel
import argparse
import sentencepiece as spm
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from collections import defaultdict

'''
LM true test Entropy
LLM`s Entropy of generated stories
Side-by-side comparison (shuffle sides)
'''

class ModelWrapper:
    def __init__(self, model, tokenizer, eos_id) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.model.eval()

    def encode(self, text):
        raise NotImplemented()

    def __call__(self, ids):
        raise NotImplemented()
    
    def decode(self, ids):
        return self.tokenizer.decode(ids[1:-1])


class MyModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer, tokenizer.eos_id())
    
    def encode(self, text):
        return [self.tokenizer.bos_id()] + self.tokenizer.encode(text)

    def __call__(self, ids):
        with torch.no_grad():
            lengths = torch.tensor([len(ids)], device=device)
            text_tensor = torch.tensor([ids], device=device)
            out = self.model(text_tensor, lengths, dtype=torch.bfloat16)
            return out[0][-1]


class GPT2XLModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer, tokenizer.eos_token_id)
    
    def encode(self, text):
        return [self.tokenizer.bos_token_id] + self.tokenizer.encode(text)
    
    def __call__(self, ids):
        with torch.no_grad():
            text_tensor = torch.tensor([ids], device=device)
            out = self.model(text_tensor)
            return out[0][0][-1]



def simple_sampler(prefix, model, min_length=100, max_length=400, top_p=0.2):
    prefix = model.encode(prefix)

    for i in range(max_length):
        logits = model(prefix)
        if i + 1 < min_length:
            logits[model.eos_id] = -torch.inf
        probs = torch.nn.functional.softmax(logits, dim=0)
        idx = probs.argsort(descending=True)
        cum_prob = 0
        j = 0
        while cum_prob < top_p or j <= 1:
            cum_prob += probs[idx[j]]
            j += 1
        idx = idx[:j]
        print(f'topp {top_p} idx', idx.shape)
        new_probs = torch.nn.functional.softmax(logits[idx], dim=0)
        final_idx = random.choices(idx, new_probs)[0].item()
        prefix.append(final_idx)
        if final_idx == model.eos_id:
            break

    return model.decode(prefix)


--n_layers=8 --embed_dim=512 --epochs=10 --n_workers=10 --mini_batch_tokens=100000 --max_seq_length=512 --batch_tokens=100000
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/final.pth"))
    parser.add_argument("--outdir", type=Path, default=Path("comparison_artifacts"))
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--checkpoint_bs_tokens", type=int, default=7000)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--topp", type=float, default=0.9)
    parser.add_argument("--min_tokens", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda:0')
    test_dataloader = get_loader("val",
                                 args.checkpoint_bs_tokens, 
                                 num_workers=args.num_workers)
    
    tokenizer = spm.SentencePieceProcessor(model_file='spm-tokenizer/spm.model')
    model = SimplerDimplerModel(len(tokenizer), args.n_layers, args.embed_dim, args.n_heads,
                                args.embed_dim * 4)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()
    model = model.to(device)
    with open("generation_prompts.json") as file:
        prompts = json.load(file)


    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=0)
        total_loss = 0
        total_items = 0
        max_length = 0
        for i, batch in tqdm(enumerate(test_dataloader)):
            text = batch['text'].to(device)
            length = batch['length'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(text, length, dtype=torch.bfloat16)
                out = out[:, :-1]
                text = text[:, 1:]
                loss = criterion(out.reshape(-1, len(tokenizer)), text.flatten())
                total_loss += loss.item()
                total_items += length.shape[0]
            if i > 100:
                break
        print(f"Checkpoint test-dataset mean loss per seq: {total_loss / total_items:.2f}")
    
    with open(args.outdir / "my-checkpoint.json", 'w') as file: 
        j = []
        modelwrapper = MyModelWrapper(model, tokenizer)
        for prompt in prompts:
            for k in range(args.generations):
                j.append(
                    simple_sampler(prompt, modelwrapper, args.min_tokens, args.max_tokens, args.topp)
                )
        json.dump(j, fp=file, indent=4)
    del model
    

    baseline_model = GPT2LMHeadModel.from_pretrained('gpt2-xl',
                                                    torch_dtype=torch.bfloat16).to(device)
    baseline_model.eval()
    baseline_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=0)
        total_loss = 0
        total_items = 0
        max_length = 0
        print(len(test_dataloader), 'test dataloader length')
        for i, batch in tqdm(enumerate(test_dataloader)):
            raw_texts = tokenizer.decode(batch['text'].tolist())
            text_encoded = [
                [baseline_tokenizer.bos_token_id] + baseline_tokenizer.encode(el) \
                + [baseline_tokenizer.eos_token_id] for el in raw_texts
            ]
            length = [len(el) for el in text_encoded]
            max_length = max(length)
            for i in range(len(text_encoded)):
                text_encoded[i] = (text_encoded[i] + [0] * (max_length - length[i]))
            text = torch.tensor(text_encoded, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                all_output = baseline_model(text)
                out = all_output[0]
                out = out[:, :-1]
                text = text[:, 1:]
                loss = criterion(out.reshape(-1, out.shape[-1]), text.flatten())
                total_loss += loss.item()
                total_items += len(length)
            if i > 100:
                break
        print(f"GPT2-XL test-dataset mean loss per seq: {total_loss / total_items:.2f}")

    with open(args.outdir / "gpt2xl-checkpoint.json", 'w') as file: 
        j = []
        modelwrapper = GPT2XLModelWrapper(baseline_model, baseline_tokenizer)
        for prompt in prompts:
            for k in range(args.generations):
                j.append(
                    simple_sampler(prompt, modelwrapper, args.min_tokens, args.max_tokens, args.topp)
                )
        json.dump(j, fp=file, indent=4)