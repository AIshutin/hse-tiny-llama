# Language Modeling


### Installation & Training 

```shell
pip3 install -r requirements.txt
./download_all.sh
./preprocessing.sh
python3 train.py  --n_layers=8 --embed_dim=512 --epochs=10 --n_workers=10 --mini_batch_tokens=100000 --max_seq_length=512 --batch_tokens=100000 
```

It's trained using 80gb A100.

### Downloading checkpoints

```shell
pip3 install -r requirements.txt
./download_all.sh
pip3 install gdown
gdown https://drive.google.com/file/d/1qhyMX7hA8niI5_fnzwUSh2Oxib2rVjpu/view?usp=sharing -O checkpoints/final.pth --fuzzy
gdown https://drive.google.com/file/d/1v7rO9-diB6cKZVKuLTMjaV4V2htJ9_1h/view?usp=sharing -O spm-tokenizer.zip --fuzzy
unzip spm-tokenizer.zip
gdown https://drive.google.com/file/d/1AglrB9GG8cmQ1IKGfg4dPX3yx_8_EQVH/view?usp=sharing -O TinyStories/val_dataset.json --fuzzy
```

### Evaluation

