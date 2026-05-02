"""
Оптимизированный evaluator для работы с vLLM.
Использует OpenAI API для батчинга и асинхронных запросов.
"""
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    print("[evaluator-vllm] Устанавливаем openai...")
    subprocess.run(["pip", "install", "-q", "openai>=1.0"], check=True)
    from openai import AsyncOpenAI, OpenAI


PATH_TRAIN = os.environ["PASSLLM_TRAIN"]
BASE_MODEL = os.environ["PASSLLM_BASE_MODEL"]
DEVICE = os.environ["DEVICE"]

# vLLM параметры
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "passllm")
VLLM_MAX_TOKENS = 20
VLLM_BATCH_SIZE = int(os.environ.get("VLLM_BATCH_SIZE", "32"))
VLLM_TIMEOUT = 300

# Конфигурация для guess_one
max_length_sysprompt = 10000
guess_one_length_buckets = [
    int(int(os.environ["PASSLLM_NUM_GUESSES"]) * 0.10),
    int(int(os.environ["PASSLLM_NUM_GUESSES"]) * 0.25),
    int(int(os.environ["PASSLLM_NUM_GUESSES"]) * 0.25),
    int(int(os.environ["PASSLLM_NUM_GUESSES"]) * 0.20),
    int(int(os.environ["PASSLLM_NUM_GUESSES"]) * 0.20),
    int(int(os.environ["PASSLLM_NUM_GUESSES"]) * 0.00),
    int(int(os.environ["PASSLLM_NUM_GUESSES"]) * 0.00)
]

NUM_GUESSES = int(os.environ["PASSLLM_NUM_GUESSES"])
verb_check_password = 20
shut_down_verbose_logs = False
EnableDump = os.environ.get("EnableDump", "500")
PASSLLM_EVAL_DUMP_PATH = os.environ.get("PASSLLM_EVAL_DUMP_PATH", "./logs")


class vLLMEvaluator:
    """Оптимизированный evaluator с vLLM для быстрой генерации паролей."""
    
    def __init__(self, api_base: str = VLLM_API_BASE, model_name: str = VLLM_MODEL_NAME):
        self.api_base = api_base
        self.model_name = model_name
        self.client = OpenAI(api_key="local", base_url=api_base)
        self.async_client = AsyncOpenAI(api_key="local", base_url=api_base)
        self._verify_connection()
    
    def _verify_connection(self):
        """Проверяет подключение к vLLM серверу."""
        try:
            models = self.client.models.list()
            print(f"[evaluator-vllm] Подключено к vLLM, доступные модели: {[m.id for m in models.data]}")
        except Exception as e:
            raise RuntimeError(f"[evaluator-vllm] Не удалось подключиться к vLLM на {self.api_base}: {e}")
    
    def generate_single(
        self,
        prompt: str,
        old_password: str,
        max_tokens: int = VLLM_MAX_TOKENS,
        num_return_sequences: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        """
        Генерирует пароли для одного старого пароля.
        
        Args:
            prompt: System prompt для модели
            old_password: Старый пароль (контекст)
            max_tokens: Максимум токенов для генерации
            num_return_sequences: Количество вариантов для генерации
            temperature: Температура (0 = детерминированно, >0 = случайно)
            top_p: Nucleus sampling
        
        Returns:
            Список сгенерированных паролей
        """
        knowledge = json.dumps({"Old password": old_password})
        suffix = "\nPassword:" if not prompt.strip().endswith("Password:") else " "
        full_prompt = prompt.strip() + "\n" + knowledge + suffix
        
        try:
            # используем beam search (num_beams) вместо sampling
            response = self.client.completions.create(
                model=self.model_name,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                best_of=num_return_sequences,
                n=1,  # vLLM не поддерживает n > 1 в режиме beam search
                stop=["\n", " "],
                timeout=VLLM_TIMEOUT,
            )
            
            guesses = []
            for choice in response.choices:
                text = choice.text.strip()
                if text:
                    # Очистка вывода
                    if text.lower().startswith("password:"):
                        text = text[9:].strip()
                    if " " in text or "\t" in text:
                        text = text.split()[0]
                    text = text[:64].strip().rstrip(".")
                    if text and text not in guesses:
                        guesses.append(text)
            
            return guesses if guesses else [""]
        
        except Exception as e:
            print(f"[evaluator-vllm] Ошибка при генерации: {e}")
            return [""]
    
    def generate_batch(
        self,
        prompt: str,
        old_passwords: List[str],
        max_tokens: int = VLLM_MAX_TOKENS,
        num_return_sequences: int = 1,
    ) -> List[List[str]]:
        """
        Генерирует пароли для батча старых паролей.
        Это основной способ для получения лучшего GPU utilization.
        
        Args:
            prompt: System prompt
            old_passwords: Список старых паролей
            max_tokens: Макс токенов
            num_return_sequences: Кол-во вариантов для генерации
        
        Returns:
            Список списков сгенерированных паролей
        """
        results = []
        
        # Батчим запросы для эффективности
        for i in range(0, len(old_passwords), VLLM_BATCH_SIZE):
            batch = old_passwords[i:i+VLLM_BATCH_SIZE]
            batch_results = []
            
            # Создаем список промптов для батча
            prompts = []
            for old_pwd in batch:
                knowledge = json.dumps({"Old password": old_pwd})
                suffix = "\nPassword:" if not prompt.strip().endswith("Password:") else " "
                full_prompt = prompt.strip() + "\n" + knowledge + suffix
                prompts.append(full_prompt)
            
            try:
                # Используем batch processing в vLLM
                # vLLM автоматически оптимизирует батчинг
                completions = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompts,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["\n", " "],
                    timeout=VLLM_TIMEOUT,
                )
                
                for completion in completions.choices:
                    text = completion.text.strip()
                    if text:
                        if text.lower().startswith("password:"):
                            text = text[9:].strip()
                        if " " in text or "\t" in text:
                            text = text.split()[0]
                        text = text[:64].strip().rstrip(".")
                    batch_results.append([text] if text else [""])
                
                results.extend(batch_results)
            
            except Exception as e:
                print(f"[evaluator-vllm] Ошибка в батче: {e}")
                results.extend([[""] for _ in batch])
        
        return results
    
    def guess_one(
        self,
        prompt_text: str,
        old_password: str,
        max_new_tokens: int = 20,
        min_new_tokens: int = 4,
        num_guesses: int = NUM_GUESSES,
        batch_size: int = 16,
        extra_guesses_ratio: float = 0.2,
        required_length: Optional[int] = None,
    ) -> List[str]:
        """
        Генерирует N угадываний пароля для одного старого пароля.
        
        Args:
            prompt_text: Промпт для модели
            old_password: Старый пароль
            max_new_tokens: Макс токенов на генерацию
            min_new_tokens: Мин токенов
            num_guesses: Количество угадываний
            batch_size: Размер батча (не используется для vLLM, но совместимо)
            extra_guesses_ratio: Дополнительные генерации для фильтрации
            required_length: Если задано, генерирует ровно такую длину
        
        Returns:
            Список из num_guesses угадываний
        """
        knowledge = json.dumps({"Old password": old_password})
        suffix = "\nPassword:" if not prompt_text.strip().endswith("Password:") else " "
        full_input = prompt_text.strip() + "\n" + knowledge + suffix
        
        guesses = []
        total_to_generate = int(num_guesses * (1 + extra_guesses_ratio))
        
        # vLLM с beam search для генерации нескольких вариантов
        try:
            # Генерируем multiple completions через несколько запросов
            # (vLLM может быть ограничен в поддержке num_return_sequences)
            for attempt in range((total_to_generate // 10) + 1):
                try:
                    response = self.client.completions.create(
                        model=self.model_name,
                        prompt=full_input,
                        max_tokens=required_length or max_new_tokens,
                        min_tokens=min_new_tokens if not required_length else required_length,
                        temperature=0.0,
                        top_p=1.0,
                        stop=["\n", " "],
                        best_of=10,  # beam search эффективнее в vLLM
                        timeout=VLLM_TIMEOUT,
                    )
                    
                    for choice in response.choices:
                        text = choice.text.strip()
                        if not text:
                            continue
                        
                        if text.lower().startswith("password:"):
                            text = text[9:].strip()
                        if " " in text or "\t" in text:
                            text = text.split()[0]
                        text = text[:64].strip().rstrip(".")
                        
                        if required_length and len(text) != required_length:
                            continue
                        
                        if text and text not in guesses:
                            guesses.append(text)
                    
                    if len(guesses) >= num_guesses:
                        break
                
                except Exception as e:
                    if attempt == 0:
                        print(f"[evaluator-vllm] Ошибка генерации: {e}")
                    break
        
        except Exception as e:
            print(f"[evaluator-vllm] Критическая ошибка: {e}")
        
        # Дополняем пустыми если не хватает
        if len(guesses) < num_guesses:
            guesses.extend([""] * (num_guesses - len(guesses)))
        
        return guesses[:num_guesses]


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Оценивает качество промпта на train.json с использованием vLLM.
    
    Args:
        program_path: Путь к файлу с промптом
    
    Returns:
        Словарь с метриками
    """
    started_at = time.time()
    dump_enabled = bool(EnableDump)
    dump_dir = PASSLLM_EVAL_DUMP_PATH
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, f"passllm_eval_dump_{int(time.time())}.json")
    
    prompt_file = Path(program_path)
    if not prompt_file.exists():
        raise ValueError(f"Файл промпта не найден: {program_path}")
    
    try:
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise ValueError(f"Ошибка при чтении промпта: {e}")
    
    prompt_length = len(prompt_text)
    
    if not PATH_TRAIN or not os.path.exists(PATH_TRAIN):
        raise ValueError("train.json не найден")
    
    with open(PATH_TRAIN, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    if isinstance(train_data, dict):
        train_data = [train_data]
    
    max_eval = NUM_GUESSES
    train_data = random.sample(train_data, min(max_eval, len(train_data)))
    print(f"[evaluate-vllm] Оцениваем на {len(train_data)} примерах")
    
    # Инициализируем evaluator с vLLM
    try:
        evaluator = vLLMEvaluator(api_base=VLLM_API_BASE, model_name=VLLM_MODEL_NAME)
    except Exception as e:
        print(f"[evaluate-vllm] Ошибка подключения к vLLM: {e}")
        print("[evaluate-vllm] Убедитесь, что vLLM сервер запущен на", VLLM_API_BASE)
        raise
    
    correct = 0
    total = len(train_data)
    dump_rows = []
    
    for idx, item in enumerate(train_data, start=1):
        old = (item.get("Knowledge") or {}).get("Old password", "")
        target = (item.get("password") or "").strip()
        
        guesses = evaluator.guess_one(
            prompt_text,
            old,
            max_new_tokens=20,
            min_new_tokens=4,
            num_guesses=NUM_GUESSES,
            batch_size=VLLM_BATCH_SIZE,
            extra_guesses_ratio=float(os.environ.get("PASSLLM_EXTRA_GUESSES_RATIO", "0.2")),
        )
        
        hit = target in guesses
        if hit:
            correct += 1
        
        if dump_enabled:
            dump_rows.append({
                "idx": idx,
                "old_password": old,
                "target_password": target,
                "hit": hit,
                "guesses": guesses,
            })
        
        if idx % verb_check_password == 0:
            print(f"[evaluate-vllm] Проверено {idx}/{total}, попадания={correct}")
    
    cracked_rate = correct / total if total > 0 else 0.0
    
    if dump_enabled:
        payload = {
            "meta": {
                "checked": total,
                "num_guesses": NUM_GUESSES,
                "prompt_length": prompt_length,
                "cracked_rate": cracked_rate,
            },
            "rows": dump_rows,
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[evaluate-vllm] Дамп сохранён: {dump_path}")
    
    elapsed = time.time() - started_at
    print(f"[evaluate-vllm] ГОТОВО: проверено={total}, cracked_rate={cracked_rate:.4f}, время={elapsed:.1f}с")
    
    return {"combined_score": cracked_rate, "cracked_rate": cracked_rate, "prompt_length": prompt_length}


if __name__ == "__main__":
    # Пример использования
    evaluator = vLLMEvaluator()
    
    prompt = "Given OLD password, predict NEW password. Output only the password, no explanation."
    old_pwd = "test123"
    
    guesses = evaluator.guess_one(prompt, old_pwd, num_guesses=10)
    print(f"Угадывания для '{old_pwd}': {guesses}")
