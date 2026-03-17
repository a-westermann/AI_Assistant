import requests

LM_STUDIO_URL = "http://localhost:1234/api/v1/chat"
MODEL = "qwen/qwen3-vl-4b"


def ask_lmstudio(prompt: str) -> dict:
    """Text-only request to LM Studio."""
    payload = {
        "model": MODEL,
        "input": prompt,
        "store": False,
    }
    r = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def ask_lmstudio_with_images(prompt: str, image_data_urls: list[str] | None = None) -> dict:
    """
    Request to LM Studio with optional images (vision input).
    image_data_urls: list of "data:image/...;base64,..." strings.
    If empty/None, falls back to text-only input.
    """
    if not image_data_urls:
        return ask_lmstudio(prompt)

    input_parts = [{"type": "message", "content": prompt}]
    for data_url in image_data_urls:
        input_parts.append({"type": "image", "data_url": data_url})

    payload = {
        "model": MODEL,
        "input": input_parts,
        "store": False,
    }
    r = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    result = ask_lmstudio("Give me 3 names for a sci-fi colony ship.")
    print(result['output'][0]['content'])
