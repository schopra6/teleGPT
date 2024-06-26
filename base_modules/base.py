import os
import torch
import inspect
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from .config import Config as cfg
from .model import GPT
from abc import ABCMeta, abstractmethod

class DDPContext:
    def __init__(self, use_ddp):
        self.use_ddp = use_ddp

    def ddp_setup(self):
        """

        This function initializes the distributed data parallel (DDP) environment
        and configures its parameters. Its purpose is to establish the environment
        for distributed training. It starts by initializing the process group with a specified backend,
        sets the CUDA device for the current process, adjusts gradient accumulation
        steps based on the number of processes, and sets a manual seed for random number generation.
        Additionally, it enables TensorFloat32 (TF32) for matrix multiplication (matmul) and CuDNN operations.
        """

        init_process_group(backend=cfg.ddp.backend)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        seed_offset = int(os.environ['RANK'])
        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    def __enter__(self):
        if self.use_ddp:
            self.ddp_setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_ddp:
            destroy_process_group()

class BaseGram(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, **kwargs):
        
        self._init_config(**kwargs)
        self.init_model()
        self.init_optimizer()
        self._init_scaler()
        self._init_ctx()
        self._update_gradient_accumulation_steps()

    def _init_config(self, **kwargs) -> None:
        """
        Initialize the configuration for the trainer.

        This method first loads the default configuration from the 'config.py' file.
        It then updates the configuration with any keyword arguments passed to the method.
        Finally, it sets the seed for the random number generator to ensure reproducibility.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        Raises:
            None
        """
        # Update the configuration with any keyword arguments passed to the method
        for key, value in kwargs.items():
            # list all subconfigurations in a dictionary
            subconfigs = {
                "gpt": cfg.gpt,
                "io_metrics": cfg.io_metrics,
                "data": cfg.data,
                "optimizer": cfg.optimizer,
                "learning_rate": cfg.learning_rate,
                "ddp": cfg.ddp,
                "system": cfg.system,
                "sampling": cfg.sampling
            }
            for subconfig_name, subconfig in subconfigs.items():
                if hasattr(subconfig, key):
                    setattr(subconfig, key, value)
                    break
            else:  # if no break, attribute was not found in any subconfig
                print(f"Warning: Invalid config key: {key} - this will be ignored.")


    def _init_ctx(self) -> None:
        """
        Initializes the context manager for mixed precision training.
        """

        ptdtype = {'float32': torch.float32,
                   'bfloat16': torch.bfloat16,
                   'float16': torch.float16}[cfg.system.dtype]
        
        self.ctx = nullcontext() if cfg.system.use_cuda else torch.amp.autocast(device_type='cuda', 
                                                                                dtype=ptdtype)

    def _update_gradient_accumulation_steps(self) -> None:
        """
        Updates the number of gradient accumulation steps based on the DDP world size.
        """

        if cfg.ddp.ddp:
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            assert cfg.data.gradient_accumulation_steps % ddp_world_size == 0
            cfg.data.gradient_accumulation_steps //= ddp_world_size

    def _init_file_paths(self) -> None:
        """
        Initializes the file paths for saving the model state and the training log.

        This method first determines the library directory based on the "cfg.io_metrics.folder" attribute.
        It then builds the file path for saving the model state and the training log.

        Args:
            None

        Returns:
            None
        """
        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(cfg.io_metrics.out_dir, file_format.format(*args))
        
        # Determine the library directory based on the "cfg.io_metrics.folder" attribute
        if cfg.io_metrics.out_dir is None:
            self.lib_dir = os.path.dirname(os.path.realpath(__file__)) # Use the current directory

        # Get the file path configurations from the '_log_build_file_path' method
        file_path_configs = self._log_build_file_path()

        # Build the file path for saving the model
        self.file_path = build_file_path(file_path_configs['file_format'], *file_path_configs['args'])

    def init_model(self) -> None:
        """
        Initializes the model for training.

        This method initializes a model for training. It supports starting from a
        pretrained model or resuming from a checkpoint. 

        If the model is initialized with CUDA support, it will be moved to the 
        appropriate device. 

        Returns:
            model (GPT): The initialized model.
        """        
        # Device setup
        self.device = int(os.environ["LOCAL_RANK"]) if cfg.ddp.ddp \
                    else torch.device('cuda' if cfg.system.use_cuda else 'cpu')

        # If specified in the configuration, resume training from a checkpoint
        if cfg.io_metrics.init_from == 'resume':
            if not cfg.ddp.ddp or self.device == 0:
                print(f"Resuming from checkpoint {cfg.io_metrics.out_dir}")
            model = self._load_model()

        # Alternatively, If specified in the configuration, initialize from a pretrained model with
        # pretrained GPT-2 weights
        elif cfg.io_metrics.init_from.startswith('gpt2'):
            if not cfg.ddp.ddp or os.environ.get('RANK', '0') == '0':
                print(f"Initializing from OpenAI GPT-2 weights {cfg.io_metrics.init_from}")
            model = self.from_pretrained(cfg.io_metrics.init_from)

        else:
            # If neither of the above options were specified, initialize a new model
            if not cfg.ddp.ddp or self.device == 0:
                print("Initializing new model")
                self._init_file_paths()

            model = GPT()
   
        # Move the model to the appropriate device
        self.model = model.to(self.device)

        # report number of parameters
        if not cfg.ddp.ddp or self.device == 0:
            print("number of parameters: %.2fM" % (self.model.get_num_params()/1e6,))

        self.model_args = dict(n_layer = cfg.gpt.n_layer,
                               n_head = cfg.gpt.n_head,
                               n_embd = cfg.gpt.n_embd,
                               block_size = cfg.gpt.block_size,
                               bias = cfg.gpt.bias,
                               vocab_size = cfg.gpt.vocab_size,
                               dropout = cfg.gpt.dropout
                               )

        if cfg.system.compile:
            self.model = torch.compile(self.model)

        # wrap model into DDP container
        if cfg.ddp.ddp:
            self.model = DDP(self.model, device_ids=[self.device])

    def init_optimizer(self) -> None:
        """
        Configures the optimizer for the GPT model.

        This method first collects all the model parameters that require gradients.
        It then groups these parameters into two groups based on their dimensionality.
        Any parameters that are 2D will have weight decay applied to them; all others will not.
        The method then creates an AdamW optimizer with the given learning rate, betas, and weight decay settings.
        The method uses the fused version of AdamW if it is available and if the device type is CUDA.

        Args:
            weight_decay (float): The weight decay (L2 penalty) to apply to the parameters.
            learning_rate (float): The learning rate for the optimizer.
            betas (tuple): The coefficients used for computing running averages of gradient and its square.
            device_type (str): The type of device to run the model on. Can be 'cpu' or 'cuda'.

        Returns:
            torch.optim.AdamW: The configured AdamW optimizer.
        """
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Group the parameters based on their dimensionality
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]  # 2D parameters will have weight decay
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]  # non-2D parameters will not have weight decay

        # Define optimizer groups with different weight decay settings
        optim_groups = [
            {'params': decay_params, 'weight_decay': cfg.optimizer.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Check if fused AdamW is available and if the device type is CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and cfg.system.use_cuda

        # Define extra arguments for the optimizer
        extra_args = dict(fused=True) if use_fused else dict()

        # Create AdamW optimizer with the given settings
        self.optimizer = torch.optim.AdamW(optim_groups, 
                                        lr=cfg.learning_rate.learning_rate, 
                                        betas=cfg.optimizer.betas, 
                                        **extra_args)
    
    def _init_scaler(self) -> None:
        """
        Initialize the scaler for mixed precision training.

        This method first checks if the device type is CUDA and if the scaler is available.
        If both conditions are met, it initializes the scaler with the default settings.

        Args:
            None

        Returns:
            torch.cuda.amp.GradScaler: The initialized scaler.
        """
        # Check if the device type is CUDA and if the scaler is available
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.system.dtype == 'float16')) \
            if cfg.system.use_cuda and torch.cuda.amp is not None else nullcontext()

    def _save_model(self, 
                    iter_num: int,
                    best_val_loss: float = None)-> None:
        """
        Save the current state of the model to a checkpoint file.

        Returns:
            None
        """
        try:
            os.makedirs(os.path.dirname(self.file_path), mode=0o755, exist_ok=True)
        except FileExistsError:
            pass

        # The following section is your provided code incorporated into this function:
        raw_model = self.model.module if cfg.ddp.ddp else self.model # unwrap DDP container if needed

        state = {
            'model': raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': self.config_to_dict()
        }

        print(f"saving checkpoint")
        torch.save(state, self.file_path)


    def _load_model(self, optimizer=None) -> None:
        """
        Load a previously saved model and optimizer state from a checkpoint file.

        Args:
            model: The model object to load the saved state into.
            optimizer (optional): The optimizer object to load the saved state into. 

        Returns:
            None

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        
        # Check if the file path exists
        if not os.path.exists(cfg.io_metrics.out_dir):
            raise FileNotFoundError(f"File path '{cfg.io_metrics.out_dir}' does not exist")
        
        # Load the state from the file path
        checkpoint = torch.load(cfg.io_metrics.out_dir, 
                                map_location=torch.device('cuda' if cfg.system.use_cuda else 'cpu'))
        
        # Get the base name of the file (excluding the directory path)
        filename = os.path.basename(cfg.io_metrics.out_dir)

        # Remove the '.state' and split by underscore
        params = filename.replace('.state', '').split('_')

        # Assign parameters to the configuration object
        cfg.io_metrics.name = str(params[0])
        cfg.gpt.block_size = int(params[1])
        cfg.gpt.vocab_size = int(params[2])  # This is typically fixed for GPT models
        cfg.gpt.n_layer = int(params[3])
        cfg.gpt.n_head = int(params[4])
        cfg.gpt.n_embd = int(params[5])
        cfg.gpt.dropout = float(params[6])
        cfg.gpt.bias = params[7] == 'True'  # Convert from string to boolean

        self.file_path = cfg.io_metrics.out_dir
        
        # Update the 'model_args', 'iter_num', 'best_val_loss', and 'config' attributes
        self.model_args = checkpoint['model_args']
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        self.config = checkpoint['config']

        # Load the model state from the checkpoint
        state_dict = checkpoint['model']

        # Fix the keys of the state dictionary
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        # Initialize and load the model state
        model = GPT()
        model.load_state_dict(state_dict)

        # If provided, load optimizer state
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        return model


    def config_to_dict(self) -> dict:
        """
        list all subconfigurations in a dictionary
        """
        subconfigs = {
            **cfg.gpt.__dict__,
            **cfg.io_metrics.__dict__,
            **cfg.data.__dict__,
            **cfg.optimizer.__dict__,
            **cfg.learning_rate.__dict__,
            **cfg.ddp.__dict__,
            **cfg.system.__dict__,
            **cfg.sampling.__dict__
        }
        
        return subconfigs