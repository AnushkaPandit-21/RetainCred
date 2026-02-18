import requests
import os

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def generate_offer_message(segment, risk, offer):

    prompt = f"""
    Write a friendly credit card offer message.
    Segment: {segment}
    Risk: {risk}
    Offer: {offer}
    Max 25 words.
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 50}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        result = response.json()

        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return f"Special offer for you: {offer}. We value your loyalty!"
    except Exception:
        return f"Special offer for you: {offer}. We value your loyalty!"
