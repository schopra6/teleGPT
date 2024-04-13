import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import tiktoken
from ._transformer import LayerNorm, Block
from ..config import Config as cfg

new_line = tiktoken.get_encoding("gpt2").encode("\n")[0]

class GPT(nn.Module):
    """
    A class that represents the GPT model.
    ...

    Attributes
    ----------
    transformer : nn.ModuleDict
        Contains the transformer blocks and layers of the GPT model.
    lm_head : nn.Linear
        The linear layer at the end of the GPT model.

    """

    def __init__(self, **kwargs):
        super().__init__()
        assert cfg.gpt.vocab_size is not None
        assert cfg.gpt.block_size is not None

        # initialize the configuration
        self._init_config(**kwargs)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.gpt.vocab_size, cfg.gpt.n_embd),
            wpe = nn.Embedding(cfg.gpt.block_size, cfg.gpt.n_embd),
            drop = nn.Dropout(cfg.gpt.dropout),
            h = nn.ModuleList([Block() for _ in range(cfg.gpt.n_layer)]),
            ln_f = LayerNorm(),
        ))
        self.lm_head = nn.Linear(cfg.gpt.n_embd, cfg.gpt.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg.gpt.n_layer))

    def _init_config(self, **kwargs):
        """
        Initialize the configuration for the trainer.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        # Update the configuration with any keyword arguments passed to the method
        for key, value in kwargs.items():
            # list all subconfigurations in a dictionary
            subconfigs = {
                "gpt": cfg.gpt,
            }
            for subconfig_name, subconfig in subconfigs.items():
                if hasattr(subconfig, key):
                    setattr(subconfig, key, value)
                    break
            else:  # if no break, attribute was not found in any subconfig
                raise ValueError(f"Invalid config key: {key}")

    def get_num_params(self, non_embedding=True):

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initializes the weights of the given PyTorch module.
        """
        # Check if the given module is a Linear layer
        if isinstance(module, nn.Linear):
            # If so, initialize the weight matrix of the Linear layer with a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # If the Linear layer has a bias term, initialize it with zeros
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Check if the given module is an Embedding layer
        elif isinstance(module, nn.Embedding):
            # If so, initialize the weight matrix of the Embedding layer with a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """

        The forward method first checks if the length of the sequence to be processed is less than or equal to the block size of the model.
        It then creates the position embeddings and sums them with the token embeddings, applying dropout after.
        The sum of the embeddings is then passed through each transformer block in sequence.
        The output is finally passed through a layer normalization.
        If targets are provided, the method calculates the cross entropy loss, else it outputs the logits for the last position only.

        Args:
            idx (torch.Tensor): A tensor containing the input sequence, 
                                with dimensions [batch_size, sequence_length].
            targets (torch.Tensor, optional): A tensor containing the target sequence, 
                                            with the same dimensions as idx. Defaults to None.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= cfg.gpt.block_size, \
        f"Cannot forward sequence of length {t}, block size is only {cfg.gpt.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # Add an extra dimension to loss
            loss = loss.unsqueeze(0)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self):
        """
        This function adjusts the model's block size to a smaller value if needed.
        It updates the block size in the model's configuration,
        trims the weights for the position embeddings,
         and adjusts the attention bias if present.

        Args:
            block_size (int): The new block size to crop the model to.

        """
        # Crop the weights for the position embeddings to the new block size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:cfg.gpt.block_size])

        # Iterate over each transformer block in the model
        for block in self.transformer.h:
            # Check if the transformer block has attention bias
            if hasattr(block.attn, 'bias'):
                # If it does, crop the attention bias to the new block size
                block.attn.bias = block.attn.bias[:,:,:cfg.gpt.block_size,:cfg.gpt.block_size]
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimates the model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        This method first calculates the number of flops per iteration using the model's configuration
        parameters and the given fwdbwd_per_iter. It then computes the MFU by comparing the estimated 
        flops with the peak flops of an A100 GPU in bfloat16 mode.

        Args:
            fwdbwd_per_iter (float): The number of forward and backward passes per iteration.
            dt (float): The time duration of the iteration in seconds.
        """
        # Number of parameters in the model
        N = self.get_num_params()

        # Unpack key parameters from the configuration
        L, H, Q, T = cfg.gpt.n_layer, cfg.gpt.n_head, cfg.gpt.n_embd // cfg.gpt.n_head, cfg.gpt.block_size

        # Estimate the number of floating point operations (flops) per token and per iteration
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Compute flops achieved per second
        flops_achieved = flops_per_iter * (1.0 / dt)

        # A100 GPU bfloat16 peak flops is 312 TFLOPS
        flops_promised = 312e12

        # Compute model flops utilization (MFU) as the ratio of achieved flops to peak flops
        mfu = flops_achieved / flops_promised

        return mfu

    @torch.no_grad()
    def sample(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates new tokens conditioned on the input sequence of tokens.

        The method takes a conditioning sequence of indices and completes the sequence
        for a given number of times, feeding the predictions back into the model each time. 
        The model should be in evaluation mode for this method to work properly. 

        Args:
            idx (torch.LongTensor): The input sequence of tokens of shape (b, t) where
                b is the batch size and t is the sequence length.
            max_new_tokens (int): The number of new tokens to generate.
            temperature (float, optional): The temperature factor to scale the output logits.
                Higher values make the outputs more random. Defaults to 1.0.
            top_k (int, optional): The number of top k tokens to consider for the final
                softmax calculation. If None, all tokens are considered. Defaults to None.

        Returns:
            torch.LongTensor: The completed sequence of tokens.
        """
        # Store the initial length of the input sequence
        initial_len = idx.size(1)
    
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it to block size
            idx_cond = idx if idx.size(1) <= cfg.gpt.block_size else idx[:, -cfg.gpt.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Scale the logits at the final step by the desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # If the last generated token is a newline, break the loop
            if idx_next.item() == new_line:
                break

        # Slice idx to include only the new tokens and not include the newline token
        idx_new = idx[:, initial_len:-1]

        return idx_new

    @torch.no_grad()
    def perplexity(self, idx, targets):
        """
        Calculates the perplexity of the model's predictions on a batch of sequences.
        """
        # Forward pass
        logits, _ = self.forward(idx, targets)

        # Calculate the cross entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        # Perplexity is the exponential of the average loss over the sequence length
        perplexity = torch.exp(loss).item()

        return perplexity