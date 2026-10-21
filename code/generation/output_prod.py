import torch
import requests, json


def load_model_transformer(model_name: str, device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()

    return tokenizer, model

def load_model_quant(repo, model_name):
    from llama_cpp import Llama

    llm = Llama.from_pretrained(
        repo_id=repo,
        filename=model_name,
        n_gpu_layers=-1,
        n_ctx=8192,  # prompt and output together: main.py fits the prompt to it (local_window)
    )

    return llm


INST = "[INST] {} [/INST]"  # the instruction format of the llama.cpp model for Mistral

def chat_inputs(prompt: str, tokenizer):
    """The prompt as one user turn of the model's chat template, tokenized; plain text when it has no template."""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                             return_tensors="pt", return_dict=True)
    return tokenizer(prompt, return_tensors="pt")


def local_window(model, tokenizer):
    """
    The context a local model holds, prompt and output together, and a counter of the tokens a prompt takes in it:
    n_ctx for llama.cpp (tokenizer None), the model's position limit for Transformers.
    """
    if tokenizer is None:
        return model.n_ctx(), lambda prompt: len(model.tokenize(INST.format(prompt).encode("utf-8")))
    return (model.config.get_text_config().max_position_embeddings,
            lambda prompt: chat_inputs(prompt, tokenizer)["input_ids"].shape[1])


def generate_letter_local(prompt: str, tokenizer, model, max_new_tokens = 2048, temperature=0.2) -> str:
    """
    Generate text using a local model: Transformers when a tokenizer is given, llama.cpp otherwise. The prompt is
    never truncated here, which would cut its end (the task and the rules): main.py fits it to local_window.
    """
    if tokenizer is not None:
        inputs = chat_inputs(prompt, tokenizer).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature} if temperature > 0 else {}))

        # the new tokens only: the prompt decoded back need not match its text character for character
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    else:
        formatted_prompt = INST.format(prompt)

        output = model(
            formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["</s>"],
            echo=False
        )

        return output["choices"][0]["text"].strip()


def generate_letter_openrouter(prompt: str, model = "anthropic/claude-opus-4.6", api_key=None, temperature = 0.2, 
                               max_tokens = 2048, reasoning_effort = None, meta: dict = None, provider: dict = None) -> str:
    """
    Generate text via the OpenRouter API. reasoning_effort None leaves reasoning to the model's default; "low" keeps
    it short. It cannot be turned off: the current Claude models refuse a request without it. meta, when given, receives the provider that served the call and its usage (tokens, cost).
    provider is OpenRouter's routing preference ({"quantizations": ["fp4"]}): by default a call goes to the cheapest
    provider, whatever precision it serves the model at.
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

    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if provider:
        payload["provider"] = provider

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=600
    )
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(f"OpenRouter API error: {data['error']}")

    if meta is not None:
        meta.update(provider=data.get("provider"), usage=data.get("usage"))
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError(f"{model} stopped at max_tokens={max_tokens} (its hidden reasoning counts against it): "
                           "the answer is truncated or empty")
    if not (choice["message"].get("content") or "").strip():  # a provider now and then ends a call with nothing in it
        raise RuntimeError(f"{model} returned an empty answer ({data.get('provider')})")
    return choice["message"]["content"]


def generate_letter(prompt: str, model, tokenizer, backend, temperature = 0.2, max_tokens = 2048, api_key = None,
                    reasoning_effort = None, meta = None, provider = None) -> str:

    if backend == "openrouter":
        return generate_letter_openrouter(prompt, model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens,
                                          reasoning_effort=reasoning_effort, meta=meta, provider=provider)

    elif backend == "local":
        return generate_letter_local(prompt, model=model, tokenizer=tokenizer, max_new_tokens=max_tokens, temperature=temperature)

    else:
        raise ValueError(f"Unknown backend: {backend}")
