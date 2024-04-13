# Telegpt
Telegpt is a project that combines the power of Generative Pretraining Transformers (GPT) with the versatility of Telegram's API to create a responsive and interactive chatbot. It utilizes a model trained on chat data to respond to messages in a human-like manner.

## Installation
You can clone the Telegpt repository and install the package using the following commands:

```bash
> cd Telegpt
> ./install.sh
```

## Requirements
ensure you have installed the dependencies listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
## Preparing Chat Data for Training

Training Telegpt requires conversational data, and the provided `prepare.py` script helps to facilitate this process. This script is responsible for processing the input text files, filtering specific phrases, removing timestamps, and encoding the text using Byte-Pair Encoding (BPE). The output is a binary file containing the processed tokens, which is ready to be consumed by the training script.


1. **Exporting Chats:** From the WhatsApp mobile application, I exported individual chats. Each chat was exported as a `.txt` file and stored in a directory named `data/`. Ensure all your chat data files are placed in a specific directory.

2. **Preprocessing Data:** I used the `prepare.py` script to preprocess the concatenated chat file:
    ```bash
    python prepare.py --folder data/
    ```
   Replace `data/` with the path to your directory if you've used a different one.

After running the script, `train.bin` and `val.bin` files will be generated in the same directory. These files contain the Byte-Pair Encoded tokens of your chat data, ready to be used for training.

## Training
Telegpt allows you to train the model on your own data. A convenient shell script named `train.sh` simplifies the process of job submission to an HPC cluster using Slurm workload manager. This script will prompt you to enter computational resources requirements, generate a temporary Slurm job submission script, and submit it.


## Inference
After training the model, you can use it to generate responses. Integrating with the Telegram API allows the model to function as a chatbot.