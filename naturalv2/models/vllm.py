from vllm import LLM, SamplingParams
from typing import List, Optional, Union
import torch
import warnings

class vLLM:

    def __init__(self,
        llm_name: str,
        download_dir: str = ".",
        llm_path: str = "",
        tokenizer_path: str = "",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_seq_len: int= 8000,
        max_gen_len: int = 1,
        max_batch_size: int = 16,
        gpu_mem_util: float = 0.9,
        seed: int = None,
        system_prompt: str = '',
        add_bos: bool = False,
        length_norm: bool = False,
        num_gpus: int = None
        ):

        self.llm_name = llm_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.max_batch_size = max_batch_size
        self.add_bos = add_bos
        self.length_norm = length_norm
        self.seed = seed
        self.system_prompt = system_prompt

        print(f"Initializing vLLM with {llm_name}...")

        if not num_gpus:
            num_gpus = torch.cuda.device_count()
            
        if not llm_path or not tokenizer_path: 
            print(f"Downloading model {llm_name}...")
            print(f"Download directory: {download_dir}")
            self.llm = LLM(
                model=self.llm_name, 
                download_dir=download_dir, 
                gpu_memory_utilization=gpu_mem_util,
                tensor_parallel_size=num_gpus
            )
        else:
            self.llm = LLM(
                model=llm_path, 
                tokenizer=tokenizer_path, 
                download_dir=download_dir, 
                gpu_memory_utilization=gpu_mem_util,
                tensor_parallel_size=num_gpus 
            )
        
        print("LLM initialized!")

        self.tokenizer = self.llm.get_tokenizer()
        self.check_bos()

        self.sampling_params = SamplingParams(
            n=1, temperature=temperature, max_tokens=max_gen_len, prompt_logprobs=0
        )
        print("Sampling params initialized!")

    def check_bos(self):
        # Check if tokenizer automatically adds BOS token; set add_bos to True if BOS token to be added manually
        test_tokens = self.tokenizer.encode("test", add_special_tokens=True)
        if self.tokenizer.bos_token_id not in test_tokens:
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token
            warnings.warn(f'Adding to the prompt the bos token: {self.tokenizer.bos_token}. If this is an eos token, this tokenizer does not have a bos token.', stacklevel=2)
            self.add_bos = True

    def input_logprob(self, prompts, length_norm=False):
        
        if self.add_bos:
            prompts = [self.tokenizer.bos_token + p for p in prompts]
        prompts = [self.system_prompt + "\n\n" + p for p in prompts]

        inputs, outputs = [], []
        for p in prompts:
            inputs += [p]
            if len(inputs) == self.max_batch_size or len(outputs) + len(inputs) == len(prompts):
                outputs += self.llm.generate(inputs, self.sampling_params)
                inputs = []
        
        # Process outputs and compute log probabilities
        logprobs = []
        for output in outputs:
            input_tokens = self.tokenizer.encode(output.prompt, add_special_tokens=True)
            logprob = sum([i[j].logprob for i, j in zip(
                output.prompt_logprobs[1:], input_tokens[1:]
            )])
            
            # Optionally normalize by sequence length
            if length_norm:
                logprob = logprob / len(input_tokens)
                
            logprobs.append(logprob)
            
        return logprobs