import os
import glob
import re
import tiktoken
import numpy as np
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Prepare chat data')
parser.add_argument('--folder', type=str, help='Directory containing the chat files')

# Parse the command line arguments
args = parser.parse_args()

# Directory containing the chat files
base_dir = args.folder
# Get a list of all .txt files in the base directory
chat_files = glob.glob(os.path.join(base_dir, "*/*_chat.txt"))

# List of phrases that if found in a line, that line will be omitted
omitted_phrases = ["image omitted",
                   "sticker omitted",
                   "GIF omitted",
                   "video omitted",
                   "document omitted",
                   "audio omitted",
                   "Contact card omitted"]

# Regular expression to match timestamps
timestamp_regex = r"\[\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2}\] "

# A list to store all tokens from all files
data_tokens = []
# Loop over all chat files
for chat_file in chat_files:
    # Open the current chat file
    with open(chat_file, 'r') as infile:
        # Read all lines from the chat file
        lines = infile.readlines()
        # Filter out the lines that contain any of the omitted phrases
        filtered_lines = [line for line in lines if not any(phrase in line for phrase in omitted_phrases)]
        # Remove timestamps from the lines
        no_timestamps = [re.sub(timestamp_regex, "", line) for line in filtered_lines]
        # Encode each line using GPT-2 Byte-Pair Encoding
        enc = tiktoken.get_encoding("gpt2")
        encoded_lines = [enc.encode_ordinary(line) for line in no_timestamps]
        # Add the encoded lines to the data_tokens list
        data_tokens.extend(encoded_lines)

# Concatenate all tokens into a single numpy array
data_tokens = np.concatenate(data_tokens)

# Determine the split index for training and validation
n = len(data_tokens)
train_idx = int(n * 0.9)  # 90% of the data for training

# Split the data into training and validation
train_tokens = data_tokens[:train_idx]
val_tokens = data_tokens[train_idx:]

# Output file paths
train_output_file = os.path.join('train.bin')
val_output_file = os.path.join('val.bin')

# Write the tokens to the binary output files
train_tokens.astype(np.uint16).tofile(train_output_file)
val_tokens.astype(np.uint16).tofile(val_output_file)

# Print the total number of tokens
print(f"train has {len(train_tokens):,} tokens")
print(f"val has {len(val_tokens):,} tokens")