import os
from typing import Literal, Optional, Union

from litellm import acompletion, atext_completion, completion, text_completion


class LM:
    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.5,
        max_tokens: int = 64,
        num_retries: int = 2,
        **kwargs,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.num_retries = num_retries

        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

    def predict(
        self, prompt: Optional[str] = None, messages: Optional[list] = None, **kwargs
    ) -> Union[list[dict], list[str]]:
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "text":
            response = text_completion(
                **self._prepare_text_completion_params(messages, **kwargs)
            )
        else:
            response = completion(
                model=self.model,
                messages=messages,
                num_retries=self.num_retries,
                **kwargs,
            )

        return self._process_response(response, **kwargs)

    async def apredict(
        self, prompt: Optional[str] = None, messages: Optional[list] = None, **kwargs
    ) -> Union[list[dict], list[str]]:
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "text":
            response = await atext_completion(
                **self._prepare_text_completion_params(messages, **kwargs)
            )
        else:
            response = await acompletion(
                model=self.model,
                messages=messages,
                num_retries=self.num_retries,
                **kwargs,
            )
        return self._process_response(response, **kwargs)

    def _prepare_text_completion_params(self, messages, **kwargs) -> dict[str, str]:
        model_names = self.model.split("/", 1)
        provider, model = (
            model_names[0] if len(model_names) > 1 else "openai",
            model_names[-1],
        )

        # Use the API key and base from the request, or from the environment.
        api_key = kwargs.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
        api_base = kwargs.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

        # Build the prompt from the messages.
        prompt = "\n\n".join([x["content"] for x in messages]) 
        if kwargs.pop("get_reponse", None):
            prompt == "\n\nBEGIN RESPONSE:"

        return {
            "model": f"text-completion-openai/{model}",
            "prompt": prompt,
            "api_key": api_key,
            "api_base": api_base,
            "num_retries": self.num_retries,
            **kwargs,
        }

    @staticmethod
    def _process_response(response, **kwargs) -> Union[list[dict], list[str]]:
        if (
            kwargs.get("logprobs") is not None
            or kwargs.get("prompt_logprobs") is not None
        ):
            outputs = [
                {
                    "text": c.message.content if hasattr(c, "message") else c["text"],
                }
                for c in response["choices"]
            ]
            if kwargs.get("logprobs") is not None:
                for idx, c in enumerate(response["choices"]):
                    outputs[idx]["logprobs"] = (
                        c.logprobs if hasattr(c, "logprobs") else c["logprobs"]
                    )

            if kwargs.get("prompt_logprobs") is not None:
                for idx, c in enumerate(response["choices"]):
                    prompt_logprobs_dicts = (
                        c.prompt_logprobs
                        if hasattr(c, "prompt_logprobs")
                        else c["prompt_logprobs"]
                    )
                    prompt_logprobs = []
                    prompt_token_ids = []
                    prompt_tokens = []
                    for prompt_logprobs_dict in prompt_logprobs_dicts[
                        1:
                    ]:  # ignore the first None token
                        key = next(iter(prompt_logprobs_dict))
                        values = prompt_logprobs_dict.get(key)

                        prompt_token_ids.append(key)
                        prompt_logprobs.append(values["logprob"])
                        prompt_tokens.append(values["decoded_token"])

                    outputs[idx]["prompt_logprobs"] = prompt_logprobs
                    outputs[idx]["prompt_token_ids"] = prompt_token_ids
                    outputs[idx]["prompt_tokens"] = prompt_tokens

        else:
            outputs = [
                c.message.content if hasattr(c, "message") else c["text"]
                for c in response["choices"]
            ]

        return outputs
