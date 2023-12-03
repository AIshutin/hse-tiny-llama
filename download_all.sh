#!/usr/bin/env bash

# Download TinyStories
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
tar -xf TinyStories_all_data.tar.gz
mkdir TinyStories
mv data*.json TinyStories
