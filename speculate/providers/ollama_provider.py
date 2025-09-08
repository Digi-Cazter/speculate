# providers/ollama_provider.py
from __future__ import annotations

import os
import requests
from typing import Optional, List, Dict, Any
import time  # added
import threading  # added


class OllamaProvider:
    """
    Ollama provider with:
      • Endpoint fallback: tries /api/chat, falls back to /api/generate on 404
      • Provider-level defaults: temperature, seed, timeout, multi-shot preference
      • Scenario-level overrides: `seed` can override provider default
      • Throttle between calls: default 1s delay (configurable via call_delay_s or OLLAMA_CALL_DELAY_S)
      • Default response format & think flag (format overridable per call; think default True)

    Env vars:
      OLLAMA_BASE_URL
      OLLAMA_API_STYLE
      OLLAMA_TIMEOUT_S
      LLM_DEFAULT_SEED
      OLLAMA_CALL_DELAY_S
      OLLAMA_DEFAULT_FORMAT   (optional default format, e.g. "json")
      OLLAMA_THINK            ("true"/"false" override default think flag)
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        timeout_s: Optional[float] = None,
        default_multi_shot: bool = True,
        multi_shot: Optional[bool] = None,
        call_delay_s: Optional[float] = None,  # added
        default_format: Optional[str] = None,  # added
        think: bool = True,                    # added
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
        self.temperature = temperature

        # provider-level default seed (scenario can override)
        env_seed = os.getenv("LLM_DEFAULT_SEED")
        self.seed: Optional[int] = int(env_seed) if (env_seed is not None and env_seed != "") else seed

        # timeout
        self.timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", str(timeout_s if timeout_s is not None else 120)))

        # style: "", "chat", "generate", "openai" (auto by default)
        self.api_style = os.getenv("OLLAMA_API_STYLE", "").strip().lower()

        # scenarios can read this to decide their default
        if multi_shot is not None:
            self.default_multi_shot = bool(multi_shot)
        else:
            self.default_multi_shot = bool(default_multi_shot)

        # Inter-call delay (throttle) default 1s
        self.call_delay_s = float(
            os.getenv(
                "OLLAMA_CALL_DELAY_S",
                str(call_delay_s if call_delay_s is not None else 1.0),
            )
        )
        self._last_call_ts: float = 0.0
        self._delay_lock = threading.Lock()
        # Default format & think flag
        self.default_format = os.getenv("OLLAMA_DEFAULT_FORMAT", default_format or "").strip() or None
        env_think = os.getenv("OLLAMA_THINK")
        if env_think is not None:
            self.think = env_think.strip().lower() in ("1", "true", "yes", "y", "on")
        else:
            self.think = bool(think)

    # --------------------------------------------------------------------- utils
    def _make_options(self, eff_seed: Optional[int], **kwargs: Any) -> Dict[str, Any]:
        """Merge temperature/seed with any extra sampler options."""
        opts: Dict[str, Any] = {"temperature": kwargs.get("temperature", self.temperature)}
        if eff_seed is not None:
            opts["seed"] = int(eff_seed)
        # Allow pass-through of some common sampler controls if provided
        for k in ("top_p", "top_k", "repeat_penalty", "mirostat", "mirostat_eta", "mirostat_tau"):
            if k in kwargs and kwargs[k] is not None:
                opts[k] = kwargs[k]
        # Allow arbitrary extra options via `options_extra` dict
        if "options_extra" in kwargs and isinstance(kwargs["options_extra"], dict):
            opts.update(kwargs["options_extra"])
        return opts

    def _post(self, path: str, json_body: Dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}{path}"
        return requests.post(url, json=json_body, timeout=self.timeout_s)

    def _flatten_chat_to_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Create a single prompt suitable for /api/generate from chat turns."""
        parts: List[str] = []
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        if history:
            for m in history:
                role = m.get("role", "user")
                content = m.get("content", "")
                role_title = "Assistant" if role == "assistant" else "User"
                parts.append(f"{role_title}: {content}\n")
        parts.append(f"User: {prompt}\nAssistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------- API endpoints
    def _call_chat(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        eff_seed: Optional[int],
        think: bool,                 # added
        fmt: Optional[str],          # added (resolved upstream)
        **kwargs: Any,
    ) -> Optional[str]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._make_options(eff_seed, **kwargs),
        }
        if fmt:
            payload["format"] = fmt
        elif self.default_format:
            payload["format"] = self.default_format
        if think:
            payload["think"] = True  # included only when True

        r = self._post("/api/chat", payload)
        if r.status_code == 404:
            return None  # endpoint not supported; signal fallback
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and isinstance(data.get("message"), dict):
            return str(data["message"].get("content", "")).strip()
        return str(data).strip()

    def _call_generate(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        eff_seed: Optional[int],
        think: bool,                 # added
        fmt: Optional[str],          # added
        **kwargs: Any,
    ) -> str:
        flat_prompt = self._flatten_chat_to_prompt(prompt, system_prompt, history)
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": flat_prompt,
            "stream": False,
            "options": self._make_options(eff_seed, **kwargs),
        }
        if fmt:
            payload["format"] = fmt
        elif self.default_format:
            payload["format"] = self.default_format
        if think:
            payload["think"] = True

        r = self._post("/api/generate", payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "response" in data:
            return str(data["response"]).strip()
        return str(data).strip()

    # ----------------------------------------------------------------- public IO
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        # Throttle (simple wall-clock delay)
        if self.call_delay_s > 0:
            with self._delay_lock:
                now = time.time()
                wait_for = self._last_call_ts + self.call_delay_s - now
                if wait_for > 0:
                    time.sleep(wait_for)
                self._last_call_ts = time.time()
        """
        Generate text from Ollama.
          - If `seed` is provided, it overrides the provider default seed.
          - `history` should be a list of {"role": "...", "content": "..."} messages
            (use None for a fresh single-shot call).
          - Extra sampler options may be provided via kwargs (e.g., top_p, top_k, options_extra dict, format="json").
        """
        eff_seed = seed if seed is not None else self.seed
        style = self.api_style
        # Resolve per-call format & think (do not leave in kwargs for sampler/options)
        fmt = kwargs.pop("format", kwargs.pop("response_format", None))
        think = bool(kwargs.pop("think", self.think))
        if style == "chat":
            res = self._call_chat(prompt, system_prompt, history, eff_seed, think, fmt, **kwargs)
            if res is None:
                raise RuntimeError("OLLAMA_API_STYLE=chat set, but /api/chat is not available on this Ollama (404).")
            return res
        if style == "generate":
            return self._call_generate(prompt, system_prompt, history, eff_seed, think, fmt, **kwargs)
        if style == "openai":
            res = self._call_chat(prompt, system_prompt, history, eff_seed, think, fmt, **kwargs)
            if res is not None:
                return res
            return self._call_generate(prompt, system_prompt, history, eff_seed, think, fmt, **kwargs)
        res = self._call_chat(prompt, system_prompt, history, eff_seed, think, fmt, **kwargs)
        if res is not None:
            return res
        return self._call_generate(prompt, system_prompt, history, eff_seed, think, fmt, **kwargs)
