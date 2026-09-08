import json
import urllib.error
import urllib.request
from typing import Any, Optional, Tuple

from google import genai
from google.genai import types

from app.config import settings


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CLAUDE_MODEL = "claude-2.1"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
HTTP_CLIENT_USER_AGENT = "ResumeFormatter/1.0 (+https://localhost)"




def _normalize_provider(provider: Optional[str]) -> str:
    provider = (provider or "gemini").strip().lower()
    if provider in ("google", "googleai", "genai"):
        return "gemini"
    if provider in ("anthropic", "claude"):
        return "claude"
    if provider in ("openai",):
        return "openai"
    if provider in ("groq",):
        return "groq"
    if provider in ("ollama", "local"):
        return "ollama"
    return provider


def _get_claude_client(api_key: str):
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for Claude provider support. "
            "Install it with 'pip install anthropic'."
        ) from exc
    return Anthropic(api_key=api_key)


def get_llm_runtime(llm_settings: Optional[dict] = None) -> Tuple[Any, str, str, str]:
    provider = _normalize_provider((llm_settings or {}).get("provider"))
    api_key = ((llm_settings or {}).get("api_key") or "").strip()
    model = ((llm_settings or {}).get("model") or "").strip()

    if provider == "gemini":
        if api_key:
            client = genai.Client(api_key=api_key)
        elif settings.GOOGLE_API_KEY:
            client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        else:
            raise ValueError(
                "No Gemini API key was provided. Supply llm_api_key in the request or set GOOGLE_API_KEY."
            )
        return client, provider, model or DEFAULT_GEMINI_MODEL, api_key

    if provider == "claude":
        if not api_key:
            raise ValueError("No Claude API key was provided. Supply llm_api_key in the request.")
        client = _get_claude_client(api_key)
        return client, provider, model or DEFAULT_CLAUDE_MODEL, api_key

    if provider == "openai":
        if not api_key:
            raise ValueError("No OpenAI API key was provided. Supply llm_api_key in the request.")
        return None, provider, model or DEFAULT_OPENAI_MODEL, api_key

    if provider == "groq":
        if not api_key:
            raise ValueError("No Groq API key was provided. Supply llm_api_key in the request.")
        return None, provider, model or DEFAULT_GROQ_MODEL, api_key

    if provider == "ollama":
        return None, provider, model or DEFAULT_OLLAMA_MODEL, api_key

    raise ValueError(
        f"Provider '{provider}' is not supported for code-driven templates. "
        "Use gemini, claude, openai, groq, or ollama."
    )


def _read_json_response(request: urllib.request.Request, timeout: int = 120) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = body
        try:
            parsed = json.loads(body)
            detail = (
                parsed.get("error", {}).get("message")
                if isinstance(parsed.get("error"), dict)
                else parsed.get("message") or body
            )
        except Exception:
            pass
        raise RuntimeError(f"LLM provider returned HTTP {exc.code}: {detail}") from exc


def _generate_openai_completion(api_key: str, model: str, system_instruction: str, prompt: str, temperature: float = 0.0, max_output_tokens: int = 32000) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": HTTP_CLIENT_USER_AGENT,
        },
    )
    body = _read_json_response(request)
    return body["choices"][0]["message"]["content"].strip()


def _generate_groq_completion(api_key: str, model: str, system_instruction: str, prompt: str, temperature: float = 0.0, max_output_tokens: int = 32000) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    request = urllib.request.Request(
        DEFAULT_GROQ_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": HTTP_CLIENT_USER_AGENT,
        },
    )
    body = _read_json_response(request)
    return body["choices"][0]["message"]["content"].strip()


def _generate_ollama_completion(model: str, system_instruction: str, prompt: str, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    request = urllib.request.Request(
        DEFAULT_OLLAMA_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    body = _read_json_response(request, timeout=240)
    message = body.get("message") or {}
    return (message.get("content") or body.get("response") or "").strip()


def generate_llm_text(
    provider: str,
    client: Any,
    model: str,
    api_key: str,
    prompt: str,
    system_instruction: str,
    temperature: float = 0.0,
    max_output_tokens: int = 32000,
) -> str:
    if provider == "gemini":
        effective_api_key = api_key or settings.GOOGLE_API_KEY

        if not effective_api_key:
            raise ValueError(
                "No Gemini API key was provided. Supply llm_api_key in the request or set GOOGLE_API_KEY."
            )

        effective_model = model or DEFAULT_GEMINI_MODEL

        print(
            f"[GEMINI] provider={provider}, "
            f"model={effective_model}, "
            f"api_key_source={'llm_settings' if api_key else 'GOOGLE_API_KEY'}, "
            f"api_key_present={bool(effective_api_key)}"
        )

        if client is None:
            client = genai.Client(api_key=effective_api_key)

        response = client.models.generate_content(
            model=effective_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return (getattr(response, "text", "") or "").strip()

    if provider == "claude":
        message = client.messages.create(
            model=model,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
        return "".join(
            getattr(block, "text", "")
            for block in getattr(message, "content", [])
            if getattr(block, "type", "") == "text"
        ).strip()

    if provider == "openai":
        return _generate_openai_completion(api_key, model, system_instruction, prompt, temperature, max_output_tokens)

    if provider == "groq":
        return _generate_groq_completion(api_key, model, system_instruction, prompt, temperature, max_output_tokens)

    if provider == "ollama":
        return _generate_ollama_completion(model, system_instruction, prompt, temperature)

    raise ValueError(
        f"Provider '{provider}' is not supported for code-driven templates. "
        "Use gemini, claude, openai, groq, or ollama."
    )


def extract_json_object(text: str) -> dict:
    """
    Parse a JSON object from model output, tolerating markdown fences or a short
    preamble. Resume templates still validate by calling json.loads here.
    """
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])
