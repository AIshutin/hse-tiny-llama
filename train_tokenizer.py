import sentencepiece as spm
from datautils import *
from random import shuffle

train_dataset, val_dataset = get_tinystories_data(PlaceholderCombiner())
corpus = [train_dataset[i] for i in range(len(train_dataset)) if len(train_dataset[i].strip()) > 1]
shuffle(corpus)
with open('TinyStories/train_tinystories_lines.txt', 'w') as file:
    for line in corpus[:300000]:
        print(line.strip(), file=file)

model = spm.SentencePieceTrainer.train(
    input='TinyStories/train_tinystories_lines.txt', 
    model_prefix='spm-tokenizer/spm', 
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
    vocab_size=4000,
    num_threads=8,
    train_extremely_large_corpus=True
)