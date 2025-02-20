import asyncio

from openai import AsyncOpenAI


class GPT:
    def __init__(
        self,
        model_name: str,
        openai_api_key_path: str,
        system_prompt: str = "",
        human_template: str = "",
        temperature: float = 0.7,
        top_p: float = 1.0,
        batch_size: int = 1,
        max_tokens: int = 16,
        seed: int = 1234,
        response_format=None,
    ):
        with open(openai_api_key_path, "r") as key_file:
            openai_api_key = key_file.read().rstrip("\n")
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_template = human_template
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.seed = seed
        self.response_format = response_format

    async def predict(self, system_prompt, user_prompt):
        if self.human_template != "":
            user_prompt = self.human_template.format(**user_prompt)
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            response_format=self.response_format,
        )
        return response.choices[0].message.content

    def get_outputs(self, system_prompt, user_prompts):
        return [
            asyncio.run(self.predict(system_prompt, user_prompts[k]))
            for k in range(len(user_prompts))
        ]
