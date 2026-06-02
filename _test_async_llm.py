import os, sys, asyncio
from dotenv import load_dotenv
load_dotenv('D:/素材/AE_LLM_RL_FOF-main/.env')
import httpx
from openai import AsyncOpenAI

async def test():
    # 使用带timeout的AsyncClient
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0))
    client = AsyncOpenAI(
        api_key=os.environ['LLM_API_KEY'],
        base_url=os.environ['LLM_BASE_URL'],
        http_client=http_client,
    )
    resp = await client.chat.completions.create(
        model=os.environ['MODEL_NAME'],
        messages=[{'role': 'user', 'content': 'Say hello'}],
        max_tokens=10
    )
    sys.stderr.write(f'RESULT: {resp.choices[0].message.content}\n')
    await http_client.aclose()

asyncio.run(test())
