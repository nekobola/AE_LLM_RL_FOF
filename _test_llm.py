import os, json, sys
from dotenv import load_dotenv
load_dotenv('D:/素材/AE_LLM_RL_FOF-main/.env')
import httpx

try:
    resp = httpx.post(
        os.environ['LLM_BASE_URL'] + '/chat/completions',
        headers={
            'Authorization': f"Bearer {os.environ['LLM_API_KEY']}",
            'Content-Type': 'application/json'
        },
        json={
            'model': os.environ['MODEL_NAME'],
            'messages': [{'role': 'user', 'content': 'Say hello'}],
            'max_tokens': 10
        },
        timeout=60.0
    )
    data = resp.json()
    with open('D:/test_api.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    sys.stderr.write('OK\n')
except Exception as e:
    sys.stderr.write(f'ERROR: {e}\n')
