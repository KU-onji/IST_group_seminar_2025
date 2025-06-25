import os
from typing import List

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel


async def call_gpt_async(prompt: str | List[str], model="gpt-4o-mini", temperature=0.5, format=None):
    def generate_request(prompt):
        if len(prompt) > 200:
            completion_tokens = 1500
        else:
            completion_tokens = 300
        if isinstance(prompt, str):
            return {
                "model": model,
                "input": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_output_tokens": completion_tokens,
            }
        else:
            context = [
                {"role": "user", "content": p} if i % 2 == 0 else {"role": "assistant", "content": p}
                for i, p in enumerate(prompt)
            ]
            return {
                "model": model,
                "input": context,
                "temperature": temperature,
                "max_output_tokens": completion_tokens,
            }

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY_YMMT"))
    if format is not None:
        try:
            res = await client.responses.parse(**generate_request(prompt), text_format=format)
            return res.output_parsed
        except BaseException as e:
            print(f"{e.__class__.__name__}: {e}")
            return {}
    else:
        res = await client.responses.create(**generate_request(prompt))
        print(res.usage.total_tokens)
        return res.choices[0].message.content


def call_gpt(prompt: str | List[str], model="gpt-4o-mini", temperature=0.5, format=None):
    def generate_request(prompt):
        if len(prompt) > 200:
            completion_tokens = 1500
        else:
            completion_tokens = 300
        if isinstance(prompt, str):
            return {
                "model": model,
                "input": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_output_tokens": completion_tokens,
            }
        else:
            context = [
                {"role": "user", "content": p} if i % 2 == 0 else {"role": "assistant", "content": p}
                for i, p in enumerate(prompt)
            ]
            return {
                "model": model,
                "input": context,
                "temperature": temperature,
                "max_output_tokens": completion_tokens,
            }

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_YMMT"))
    if format is not None:
        try:
            res = client.responses.parse(**generate_request(prompt), text_format=format)
            return res.output_parsed
        except BaseException as e:
            print(f"{e.__class__.__name__}: {e}")
            return {}
    else:
        res = client.responses.create(**generate_request(prompt))
        print(res.usage.total_tokens)
        return res.choices[0].message.content


class Hypothesis(BaseModel):
    hypothesis: str
    score: float


class Thought(BaseModel):
    current_observation: List[str]
    possible_hypotheses: List[Hypothesis]
    next_question: str | None
    is_terminal: bool
