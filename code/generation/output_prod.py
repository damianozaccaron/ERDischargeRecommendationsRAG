import torch
import requests, json


def load_model_transformer(model_name: str, device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()

    return tokenizer, model

def load_model_quant(repo, model_name):
    from llama_cpp import Llama

    llm = Llama.from_pretrained(
        repo_id=repo,
        filename=model_name,
        n_gpu_layers=-1,
        n_ctx=8192,
    )

    return llm


def generate_letter_local(prompt: str, tokenizer, model, max_new_tokens = 2048, temperature=0.2) -> str:
    """
    Generate text using a local model.
    """
    if tokenizer is not None:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                echo=False)

        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True)

        return decoded[len(prompt):].strip()
    
    else:
        formatted_prompt = f"[INST] {prompt} [/INST]"

        output = model(
            formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["</s>"],
            echo=False
        )

        return output["choices"][0]["text"].strip()


def generate_letter_openrouter(prompt: str, model = "anthropic/claude-opus-4.6", api_key=None, temperature = 0.2, 
                               max_tokens = 2048, use_reasoning = None) -> str:
    """
    Generate text via the OpenRouter API. use_reasoning None leaves reasoning to the model's default, which for some
    models is on: its hidden tokens then count against max_tokens and can leave the answer truncated or empty.
    """

    if not api_key:
        raise ValueError("No API key provided.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if use_reasoning is not None:
        payload["reasoning"] = {"enabled": use_reasoning}

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=120
    )
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(f"OpenRouter API error: {data['error']}")
    
    return data["choices"][0]["message"]["content"]


def generate_letter(prompt: str, model, tokenizer, backend, temperature = 0.2, max_tokens = 2048, api_key = None,
                    use_reasoning = None) -> str:

    if backend == "openrouter":
        return generate_letter_openrouter(prompt, model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens,
                                          use_reasoning=use_reasoning)

    elif backend == "local":
        return generate_letter_local(prompt, model=model, tokenizer=tokenizer, max_new_tokens=max_tokens, temperature=temperature)

    else:
        raise ValueError(f"Unknown backend: {backend}")
