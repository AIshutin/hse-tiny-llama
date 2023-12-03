import torch
from datautils import *
from optimutils import *
from model import SimplerDimplerModel
import argparse
import sentencepiece as spm
from pathlib import Path
import wandb
from time import sleep
from torch.cuda.amp import GradScaler
import math


def inefficient_generation(prefix: str, tokenizer, model, max_length=100, dtype=None):
    words = [tokenizer.bos_id()] + tokenizer.encode(prefix)
    while len(words) < max_length and words[-1] != tokenizer.eos_id():
        text_tensor = torch.tensor([words], device=device)
        lengths = torch.tensor([len(words)], device=device)
        out = model(text_tensor, lengths, dtype=dtype)
        out_id = out.max(dim=-1).indices[0, -1]
        words.append(out_id.item())
    return tokenizer.decode(words)


def calc_parameters(model):
    return sum([el.numel() for el in model.parameters() if el.requires_grad])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train script")
    parser.add_argument("--mini_batch_tokens", type=int, default=25000)
    parser.add_argument("--batch_tokens", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--prompt_file", type=Path, default=Path("./generation_prompts.json"))
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=1000000)
    parser.add_argument("--n_workers", default=4, type=int)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor(model_file='spm-tokenizer/spm.model')
    device = torch.device('cuda:0')
    model = SimplerDimplerModel(
        len(tokenizer), n_layers=args.n_layers, embed_dim=args.embed_dim, 
        n_heads=args.n_heads, ffw_size=4 * args.embed_dim, dropout=args.dropout, 
    ).to(device)
    #print(model)
    #for layer in model.transformer.layers:
    #    layer.self_attn.dropout = 0
    #print('MODEL_SIZE', calc_parameters(model))
    if args.compile:
        model = torch.compile(model)

    criterion = LMCriterion()

    train_loader = get_loader("train", 
                              max_seq_length=args.max_seq_length,
                              num_workers=args.n_workers,
                              mini_batch_tokens=args.mini_batch_tokens)
    val_loader   = get_loader("tiny_val", 
                              num_workers=args.n_workers,
                              mini_batch_tokens=args.mini_batch_tokens)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    lr_scheduler = get_scheduler(
        optimizer,
        args.lr,
        steps_per_epoch=math.ceil(1.01 * len(train_loader) * args.mini_batch_tokens / args.batch_tokens),
        epochs=args.epochs
    )

    run = wandb.init(
        project="dl-2-bhw1",
        config=args
    )
    wandb.run.log_code(".")
    print('MODEL_SIZE', calc_parameters(model))
    print(model)

    
    with open(args.prompt_file) as file:
        prompts = json.load(file)


    def step():
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

    scaler = GradScaler()
    total_tokens = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_tokens = 0
        current_tokens = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"train {1 + epoch}")):
            text = batch['text'].to(device)
            length = batch['length'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(text, length, dtype=torch.bfloat16)
                loss = criterion(out, text)
            train_loss += loss.item() * length.sum().item()
            train_tokens += length.sum().item()
            train_correct += (out.max(dim=-1).indices[:, :-1] == text[:, 1:]).int().sum().item()

            current_tokens += length.sum().item()
            scaler.scale(loss * length.sum().item() / args.batch_tokens).backward() 

            if current_tokens >= args.batch_tokens:
                step()
                current_tokens = 0
        if current_tokens != 0:
            step()
        train_loss_per_token = train_loss / train_tokens
        train_acc_per_token = train_correct / train_tokens
        total_tokens += train_tokens
        print(f"Train loss: {train_loss_per_token:.4f}. Train acc: {train_acc_per_token:.4f}")
        print(f"Train tokens: {total_tokens / 1e6:.2f}M")

        model.eval()
        val_loss = 0
        val_correct = 0
        val_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"val {1 + epoch}")):
                text = batch['text'].to(device)
                length = batch['length'].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(text, length)
                    loss = criterion(out, text)
                val_loss += loss.item() * length.sum().item()
                val_tokens += length.sum().item()
                val_correct += (out.max(dim=-1).indices[:, :-1] == text[:, 1:]).int().sum().item()
        
        val_loss_per_token = val_loss    / val_tokens
        val_acc_per_token  = val_correct / val_tokens
        print(f"Val loss: {val_loss_per_token:.4f}. Val acc: {val_acc_per_token:.4f}")

        texts = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                for p in prompts:
                    texts.append(inefficient_generation(p, tokenizer, 
                                                        model, dtype=torch.bfloat16))
                    print(texts[-1])

        data = list(zip(prompts, texts))
        print(data)
        table = wandb.Table(columns=["prefix", "generated"], data=data)
        
        wandb.log({
            "train_loss": train_loss_per_token,
            "train_acc": train_acc_per_token,
            "val_loss": val_loss_per_token,
            "val_acc": val_acc_per_token,
            "argmax_generation": table,
            "total_tokens": total_tokens,
            "lr": lr_scheduler.get_last_lr()[0]
        })

        
        if (epoch + 1) % args.save_epochs == 0 or epoch + 1 == args.epochs:
            state_dict = {
                "model": model.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "optim": optimizer.state_dict()
            }
            torch.save(state_dict, f"checkpoints/model-ep{epoch}-val_loss-{val_loss_per_token:.5f}.pt")