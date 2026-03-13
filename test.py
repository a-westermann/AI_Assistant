import requests

LM_STUDIO_URL = "http://localhost:1234/api/v1/chat"
MODEL = "qwen/qwen3-vl-4b"  


def ask_lmstudio(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "input": prompt,
        "store": False,
    }

    r = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    result = ask_lmstudio("Give me 3 names for a sci-fi colony ship.")
    print(result['output'][0]['content'])
