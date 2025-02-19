import warnings

import numpy as np
import torch
from scipy.special import softmax
from vllm import LLM, SamplingParams


class vLLM:
    def __init__(
        self,
        model_name: str,
        download_dir: str = ".",
        llm_path: str = "",
        tokenizer_path: str = "",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_seq_len: int = 8000,
        max_gen_len: int = 1,
        batch_size: int = 16,
        gpu_mem_util: float = 0.9,
        seed: int = None,
        system_prompt: str = "",
        add_bos: bool = False,
        length_norm: bool = False,
        num_gpus: int = None,
    ):
        self.model_name = model_name
        self.download_dir = download_dir
        self.llm_path = llm_path
        self.tokenizer_path = tokenizer_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.gpu_mem_util = gpu_mem_util
        self.add_bos = add_bos
        self.length_norm = length_norm
        self.seed = seed
        self.system_prompt = system_prompt
        self.num_gpus = num_gpus

    def load_model(self):
        print(f"Initializing vLLM with {self.model_name}...")

        if not self.num_gpus:
            self.num_gpus = torch.cuda.device_count()

        if not self.llm_path or not self.tokenizer_path:
            print(f"Downloading model {self.model_name}...")
            print(f"Download directory: {self.download_dir}")
            self.llm = LLM(
                model=self.model_name,
                download_dir=self.download_dir,
                gpu_memory_utilization=self.gpu_mem_util,
                tensor_parallel_size=self.num_gpus,
            )
        else:
            self.llm = LLM(
                model=self.llm_path,
                tokenizer=self.tokenizer_path,
                download_dir=self.download_dir,
                gpu_memory_utilization=self.gpu_mem_util,
                tensor_parallel_size=self.num_gpus,
            )

        print("LLM initialized!")

        self.tokenizer = self.llm.get_tokenizer()
        self.check_bos()

        self.sampling_params = SamplingParams(
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_gen_len,
            prompt_logprobs=0,
        )
        print("Sampling params initialized!")

    def check_bos(self):
        # Check if tokenizer automatically adds BOS token; set add_bos to True if BOS token to be added manually
        test_tokens = self.tokenizer.encode("test", add_special_tokens=True)
        if self.tokenizer.bos_token_id not in test_tokens:
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token
            warnings.warn(
                f"Adding to the prompt the bos token: {self.tokenizer.bos_token}. If this is an eos token, this tokenizer does not have a bos token.",
                stacklevel=2,
            )
            self.add_bos = True

    def input_logprob(self, prompts, length_norm=False):
        if self.add_bos:
            prompts = [self.tokenizer.bos_token + p for p in prompts]
        prompts = [self.system_prompt + "\n\n" + p for p in prompts]

        inputs, outputs = [], []
        for p in prompts:
            inputs += [p]
            if len(inputs) == self.batch_size or len(outputs) + len(inputs) == len(
                prompts
            ):
                outputs += self.llm.generate(inputs, self.sampling_params)
                inputs = []

        # Process outputs and compute log probabilities
        logprobs = []
        for output in outputs:
            input_tokens = self.tokenizer.encode(output.prompt, add_special_tokens=True)
            logprob = sum(
                [
                    i[j].logprob
                    for i, j in zip(output.prompt_logprobs[1:], input_tokens[1:])
                ]
            )

            # Optionally normalize by sequence length
            if length_norm:
                logprob = logprob / len(input_tokens)

            logprobs.append(logprob)

        return logprobs

    def compute_input_probs(self, X, options):
        X_repeat = [x for x in X for _ in range(len(options))]
        options_repeat = options * len(X)
        inp = []
        for x, y in zip(X_repeat, options_repeat):
            inp += [x + y]
        logprobs = self.input_logprob(inp)
        logprobs = np.array(logprobs).reshape((len(X), len(options)))
        probs = softmax(logprobs, axis=1)
        sample_idx = [np.random.choice(len(prob), p=prob) for prob in probs]
        max_idx = np.argmax(probs, axis=1)
        return probs, sample_idx, max_idx
