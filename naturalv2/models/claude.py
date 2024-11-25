import asyncio
from anthropic import AsyncAnthropic

class Claude:
    def __init__(self,
                 model_name: str,
                 anthropic_api_key_path: str,
                 system_template: str='',
                 human_template: str='',
                 temperature: float=0.7,
                 top_p: float=1.0,
                 max_tokens: int=16,
                 seed: int = 1234,):
        
        key_file = open(anthropic_api_key_path, "r") 
        anthropic_api_key = key_file.read().rstrip('\n')
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.model_name = model_name
        self.system_template = system_template
        self.user_template = human_template
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed

    async def predict(self, system_prompt, user_prompt):

        response = await self.client.messages.create(
                                model = self.model_name,
                                system = system_prompt,
                                messages=[
                                    { "role": "user",
                                    "content": user_prompt } ],
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                top_p=1,
        )
        return response.content[0].text

    def get_outputs(self, system_prompt, user_prompts):
        return [asyncio.run(self.predict(system_prompt, user_prompts[k])) for k in range(len(user_prompts))]
