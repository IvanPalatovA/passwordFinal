"""Targeted evaluator: one guess per train sample (Old password -> password), cracked rate on train.json."""
import os
import json
from pathlib import Path
from typing import Dict, Any
import torch
import random
import time
import multiprocessing as mp
import gc
import subprocess
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PATH_TRAIN = os.environ["PASSLLM_TRAIN"]
PATH_ADAPTER = os.environ["PASSLLM_ADAPTER_126_CSDN"]
BASE_MODEL = os.environ["PASSLLM_BASE_MODEL"]
DEVICE = os.environ["DEVICE"]

# кол-во предложенных пар - старый пароль - новый пароль
max_check_example = 50


# ограничение входного промпта для passllm
max_length_sysprompt = 3000


# кол-во генераций пароля на один target-пароль
NUM_GUESSES = int(os.environ["PASSLLM_NUM_GUESSES"])

# дополнительная в % генерация для лучевого поиска
os.environ["PASSLLM_EXTRA_GUESSES_RATIO"] = "0.2"

max_time_generate_passqwords = 100

# кол-во параллельных вычислений пароля
parallel_generations = int(os.environ["PASSLLM_NUM_GUESSES"])

os.environ["PASSLLM_EVAL_MAX_SAMPLES"] = str(max_check_example)

_cached_model = None
_cached_tokenizer = None
GPU_LOG_EVERY_N_REQUESTS = int(os.environ.get("PASSLLM_GPU_LOG_EVERY_N_REQUESTS", "10"))
_NVIDIA_SMI_PATH = shutil.which("nvidia-smi")

def _is_main_process() -> bool:
    try:
        return mp.current_process().name == "MainProcess"
    except Exception:
        return True

def _log_gpu_stats(idx: int, total: int):
    if GPU_LOG_EVERY_N_REQUESTS <= 0:
        return
    if idx % GPU_LOG_EVERY_N_REQUESTS != 0 and idx != total:
        return
    if not torch.cuda.is_available():
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mb = int(free_bytes / (1024 * 1024))
    total_mb = int(total_bytes / (1024 * 1024))
    used_mb = total_mb - free_mb
    if _NVIDIA_SMI_PATH:
        try:
            out = subprocess.check_output(
                [
                    _NVIDIA_SMI_PATH,
                    "--query-gpu=utilization.gpu,memory.free,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            ).strip().splitlines()[0]
            util, mem_free, mem_used, mem_total = [x.strip() for x in out.split(",")]
            print(f"[gpu] {idx}/{total} util={util}% vram_free={mem_free}MB vram_used={mem_used}MB vram_total={mem_total}MB")
            return
        except Exception:
            pass
    print(f"[gpu] {idx}/{total} util=NA vram_free={free_mb}MB vram_used={used_mb}MB vram_total={total_mb}MB")

def _load_model():
    global _cached_model, _cached_tokenizer
    if _cached_model is not None and _cached_tokenizer is not None:
        return _cached_model, _cached_tokenizer
    # скачивание токенизатора
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # скачивание модели
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map=DEVICE, torch_dtype=torch.float16, trust_remote_code=True)
    os.makedirs(PATH_ADAPTER, exist_ok=True)
    # скачивание надстройки для модели (+ распаковка zip)
    import zipfile
    import urllib.request
    ZENODO_ZIP_URL = "https://zenodo.org/records/15612295/files/Available%20artifacts%20for%20USENIX%20Security%202025%20%23772-v1.zip?download=1"
    os.makedirs("./temp", exist_ok=True)
    ZIP_PATH = "./temp/passllm_artifact.zip"
    if not os.path.exists(os.path.join(PATH_ADAPTER, "adapter_config.json")):
        urllib.request.urlretrieve(ZENODO_ZIP_URL, ZIP_PATH)

        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            for member in zf.namelist():
                if "/checkpoints/126_csdn_disQwen0.5B/" in member:
                    zf.extract(member, ".")

        extracted_root = os.path.join(".", "Available artifacts for USENIX Security 2025 #772-v1", "checkpoints", "126_csdn_disQwen0.5B")

        if os.path.exists(extracted_root):
            os.makedirs(os.path.dirname(PATH_ADAPTER), exist_ok=True)
            os.rename(extracted_root, PATH_ADAPTER)

    model = PeftModel.from_pretrained(model, PATH_ADAPTER, is_trainable=False)
    model = model.merge_and_unload()
    model.eval()
    _cached_model, _cached_tokenizer = model, tokenizer
    return _cached_model, _cached_tokenizer

def guess_one(prompt_text: str, old_password: str, model, tokenizer, max_new_tokens, num_guesses, batch_size, extra_guesses_ratio: float):
    knowledge = json.dumps({"Old password": old_password})
    suffix = "\nPassword:" if not prompt_text.strip().endswith("Password:") else " "
    full_input = prompt_text.strip() + "\n" + knowledge + suffix
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=max_length_sysprompt).to(model.device)

    guesses = []
    total_guesses = int(num_guesses * (1 + extra_guesses_ratio))
    cycles = total_guesses // batch_size
    remainder = total_guesses % batch_size
    batches = [batch_size] * cycles
    if remainder > 0:
        batches.append(remainder)

    try:
        stop_tokens = [tokenizer.eos_token_id] + tokenizer.encode("\n", add_special_tokens=False) + tokenizer.encode(" ", add_special_tokens=False)
        with torch.inference_mode():
            for current_batch in batches:
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=current_batch, num_return_sequences=current_batch, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=stop_tokens, max_time=max_time_generate_passqwords)
                for i in range(out.shape[0]):
                    generated = tokenizer.decode(out[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    for line in generated.splitlines():
                        guess = line.strip()
                        if not guess:
                            continue
                        if guess.lower().startswith("password:"):
                            guess = guess[9:].strip()
                        if " " in guess or "\t" in guess:
                            guess = (guess.split()[0] if guess.split() else guess)
                        guess = (guess[:64].strip() if guess else "")
                        guess = guess.rstrip('.')
                        if guess and guess not in guesses:
                            guesses.append(guess)
                        break
                del out
                if len(guesses) >= num_guesses:
                    break
    finally:
        # Явное удаление больших тензоров и очистка кэша CUDA для предотвращения утечек
        if 'out' in locals():
            del out
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    if len(guesses) < num_guesses:
        guesses.extend([""] * (num_guesses - len(guesses)))
    return guesses[:num_guesses]

def evaluate(program_path: str) -> Dict[str, Any]:
    started_at = time.time()
    dump_enabled = bool(os.environ["EnableDump"])
    dump_preview_count = int(os.environ.get("PASSLLM_EVAL_DUMP_PREVIEW_COUNT"))
    dump_dir = os.environ.get("PASSLLM_EVAL_DUMP_PATH")
    if not dump_dir:
        dump_dir = "./logs"
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, f"passllm_eval_dump_{int(time.time())}.json")

    prompt_file = Path(program_path)
    if not prompt_file.exists():
        raise ValueError("Нет стартового промпта - ОШИБКА!!!!");
        # return {"combined_score": 0.0, "cracked_rate": 0.0, "prompt_length": 0, "error": "Prompt file not found"}
    try:
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise ValueError("ОШИБКА при чтении стартового промпта");
        # return {"combined_score": 0.0, "cracked_rate": 0.0, "prompt_length": 0, "error": str(e)}
    prompt_length = len(prompt_text)
    if not PATH_TRAIN or not os.path.exists(PATH_TRAIN):
        raise ValueError("ОШИБКА - не существует train.json");
        # return {"combined_score": 0.0, "cracked_rate": 0.0, "prompt_length": prompt_length, "error": "train.json not found"}
    with open(PATH_TRAIN, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    if isinstance(train_data, dict):
        train_data = [train_data]
    max_eval = int(os.environ["PASSLLM_EVAL_MAX_SAMPLES"])
    train_data = random.sample(train_data, max_eval)
    print(f"[evaluate] to_check={len(train_data)} (max_check_example={max_check_example})")
    try:
        model, tokenizer = _load_model()
    except Exception as e:
        raise ValueError("ОШИБКА - не удалось загрузить model и tokenizator");
        # return {"combined_score": 0.0, "cracked_rate": 0.0, "prompt_length": prompt_length, "error": str(e)}
    correct = 0
    total = len(train_data)
    dump_rows = []
    shown = 0

    for idx, item in enumerate(train_data, start=1):
        old = (item.get("Knowledge")).get("Old password")
        target = (item.get("password")).strip()
        guesses = guess_one(
            prompt_text,
            old,
            model,
            tokenizer,
            max_new_tokens=20,
            num_guesses=NUM_GUESSES,
            batch_size=parallel_generations,
            extra_guesses_ratio=float(os.environ.get("PASSLLM_EXTRA_GUESSES_RATIO")),
        )
        hit = target in guesses
        if hit:
            correct += 1

        if dump_enabled:
            dump_rows.append(
                {
                    "idx": idx,
                    "old_password": old,
                    "target_password": target,
                    "hit": hit,
                    "guesses": guesses,
                }
            )

        if (idx % 10 == 0):
            print(f"[evaluate] checked {idx}/{total}, hits={correct}")
            print(f"old password is {old}")
            print(f"new password is {target}")
            if dump_enabled and shown < dump_preview_count:
                for g in guesses[:NUM_GUESSES]:
                    print("  " + g)
                shown += 1
        _log_gpu_stats(idx, total)

    cracked_rate = correct / max_eval

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
        print(f"[evaluate] dump saved: {dump_path}")

    # Keep the model cached in worker processes to avoid reloading weights.
    # In the main (notebook/kernel) process, drop cache to avoid a second large resident model.
    if _is_main_process():
        global _cached_model, _cached_tokenizer
        _cached_model = None
        _cached_tokenizer = None
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed = time.time() - started_at
    print(f"[evaluate] DONE: checked={total}, cracked_rate={cracked_rate:.4f}, elapsed_sec={elapsed:.1f}")
    return {"combined_score": cracked_rate, "cracked_rate": cracked_rate, "prompt_length": prompt_length}
