"""
scripts/llm_client.py
----------------------
One place to get an LLM client. Picks a provider and returns a ready
client plus the model name to call it with.

Providers
    openai  OpenAI API                      (default)
    azure   Azure OpenAI                    (fine-tuned deployment)
    local   any OpenAI-compatible server    (vLLM, LM Studio, Ollama, TGI)

Selection order:
    1. the `provider` argument, if given
    2. use_azure=True  -> "azure"   (kept so existing callers keep working)
    3. $LLM_PROVIDER
    4. "openai"

Every provider is called through chat.completions, which is the one
endpoint all three implement. Local servers do not implement OpenAI's
Responses API, which is why the OpenAI path no longer uses it either.

Usage:
    from llm_client import chat
    text = chat(messages, temperature=0.3)             # provider from env
    text = chat(messages, provider="local")            # forced
    text = chat(messages, use_azure=True)              # legacy callers
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROVIDERS = ("openai", "azure", "local")

_clients: dict = {}


def resolve_provider(provider: str = None, use_azure: bool = False) -> str:
    """Decide which provider to use. See selection order in module docstring."""
    if provider:
        name = provider.lower()
    elif use_azure:
        name = "azure"
    else:
        name = os.environ.get("LLM_PROVIDER", "openai").lower()

    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Set LLM_PROVIDER to one of: "
            f"{', '.join(PROVIDERS)}"
        )
    return name


def get_client(provider: str = None, use_azure: bool = False):
    """Return (client, model_name) for the selected provider. Cached per provider."""
    name = resolve_provider(provider, use_azure)
    if name not in _clients:
        _clients[name] = _build(name)
    return _clients[name]


def _build(name: str):
    if name == "openai":
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", "")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file, or set "
                "LLM_PROVIDER=local to use a local server instead."
            )
        if not model:
            raise EnvironmentError("OPENAI_MODEL is not set. Add it to your .env file.")
        return OpenAI(api_key=api_key), model

    if name == "azure":
        from openai import AzureOpenAI

        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not api_key or not endpoint:
            raise EnvironmentError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in "
                "your .env file."
            )
        model = os.environ.get("AZURE_FINETUNED_MODEL", "") or os.environ.get(
            "AZURE_BASE_MODEL", ""
        )
        if not model:
            raise EnvironmentError(
                "Set AZURE_FINETUNED_MODEL (or AZURE_BASE_MODEL) to a deployment name."
            )
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        return client, model

    # local
    from openai import OpenAI

    base_url = os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("LOCAL_MODEL", "")
    if not model:
        raise EnvironmentError(
            "LOCAL_MODEL is not set. Add it to your .env file -- use the id "
            f"reported by {base_url}/models."
        )
    # local servers ignore the key, but the SDK requires a non-empty one
    api_key = os.environ.get("LOCAL_API_KEY", "") or "not-needed"
    return OpenAI(api_key=api_key, base_url=base_url), model


def chat(
    messages: list,
    temperature: float = 0.3,
    provider: str = None,
    use_azure: bool = False,
    model: str = None,
    **kwargs,
) -> str:
    """
    Send messages, return the reply text. Extra kwargs pass through to the API.

    `model` overrides the provider's default -- used where a call needs a
    different model than the configured one (e.g. score_reliability's
    reconstruction step wants a general model, not the fine-tuned one).

    `temperature=None` omits the parameter, for models that reject it.
    """
    client, default_model = get_client(provider, use_azure)
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(
        model=model or default_model,
        messages=messages,
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip()


def active_model(provider: str = None, use_azure: bool = False) -> str:
    """Model name for the selected provider, for logging. Never raises on an unset key."""
    try:
        return f"{resolve_provider(provider, use_azure)} — {get_client(provider, use_azure)[1]}"
    except EnvironmentError as e:
        return f"{resolve_provider(provider, use_azure)} — unconfigured ({e})"


if __name__ == "__main__":
    # Self-check: provider selection only, no network calls.
    assert resolve_provider("local") == "local"
    assert resolve_provider(use_azure=True) == "azure"
    assert resolve_provider() == os.environ.get("LLM_PROVIDER", "openai").lower()
    assert resolve_provider("azure", use_azure=False) == "azure"  # explicit arg wins
    try:
        resolve_provider("nope")
        raise AssertionError("expected ValueError for unknown provider")
    except ValueError:
        pass
    print("provider selection OK")
    print("active:", active_model())
