import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 1. Unified LLM Client
# =========================
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFChatClient:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype="auto",
        trust_remote_code: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        do_sample: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.usage_history = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
                if text.startswith("json"):
                    text = text[4:].strip()
                elif text.startswith("python"):
                    text = text[6:].strip()
        return text.strip()

    def generate(
        self,
        messages,
        max_new_tokens: int = None,
        temperature: float = None,
        do_sample: bool = None,
    ) -> str:
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages must be a non-empty list")
        self.reset_usage()
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        temperature = self.temperature if temperature is None else temperature
        do_sample = self.do_sample if do_sample is None else do_sample

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = model_inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]

        prompt_tokens = input_len
        completion_tokens = output_ids.shape[0]
        total_tokens = prompt_tokens + completion_tokens

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        self.usage_history.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })

        output_text = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )

        del model_inputs, generated_ids, output_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self._strip_code_fence(output_text)

    def get_usage(self):
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "num_calls": len(self.usage_history),
            "usage_history": self.usage_history,
        }

    def reset_usage(self):
        self.usage_history = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def close(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


