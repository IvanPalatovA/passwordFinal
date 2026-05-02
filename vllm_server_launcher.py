"""
vLLM сервер для ускорения генерации паролей.
Обеспечивает лучше GPU utilization через батчинг и KV-cache.
"""
import os
import subprocess
import time
import urllib.request
import json
import sys


def launch_vllm_server(
    model_name: str,
    adapter_path: str,
    device: str = "cuda",
    gpu_memory_fraction: float = 0.9,
    max_model_len: int = 2048,
    enforce_eager: bool = False,
    port: int = 8000,
    max_num_seqs: int = 256,
    disable_log_stats: bool = False,
):
    """
    Запускает vLLM сервер с заданными параметрами.
    
    Args:
        model_name: Имя модели (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        adapter_path: Путь к LoRA адаптеру
        device: "cuda" или "cpu"
        gpu_memory_fraction: Доля GPU памяти для использования
        max_model_len: Максимальная длина контекста
        enforce_eager: Использовать eager execution (медленнее но стабильнее)
        port: Порт для API
        max_num_seqs: Максимум параллельных sequences (важно для батчинга)
        disable_log_stats: Отключить статистику
    """
    
    # Проверяем установку vLLM
    try:
        import vllm
        print(f"[vllm-server] vLLM версия: {vllm.__version__}")
    except ImportError:
        print("[vllm-server] vLLM не найден, устанавливаем...")
        subprocess.run(
            ["pip", "install", "-q", "vllm"],
            check=True
        )
        import vllm
        print(f"[vllm-server] vLLM установлен: {vllm.__version__}")
    
    # Параметры для vLLM
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--enable-lora",
        "--gpu-memory-utilization", str(gpu_memory_fraction),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--enable-prefix-caching",  # Кэширование префиксов (помогает для похожих промптов)
    ]
    
    # LoRA адаптер
    if adapter_path and os.path.exists(adapter_path):
        cmd.extend(["--lora-modules", f"passllm={adapter_path}"])
        print(f"[vllm-server] LoRA адаптер: {adapter_path}")
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    if disable_log_stats:
        cmd.append("--disable-log-stats")
    
    # Совместимый с текущими версиями vLLM флаг отключения логов запросов.
    cmd.append("--no-enable-log-requests")
    
    print("[vllm-server] Параметры:")
    for i in range(0, len(cmd), 2):
        if i+1 < len(cmd):
            print(f"  {cmd[i]} {cmd[i+1]}")
    
    print(f"[vllm-server] Запуск сервера на порту {port}...")
    
    # Запускаем сервер
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Ждем инициализации
    api_url = f"http://127.0.0.1:{port}/v1/models"
    max_retries = 60
    for attempt in range(max_retries):
        try:
            response = urllib.request.urlopen(
                urllib.request.Request(api_url, method="GET"),
                timeout=2
            )
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"[vllm-server] ✓ Сервер готов! Модели: {len(data['data'])}")
                for model in data['data']:
                    print(f"  - {model['id']}")
                return process, api_url.replace("/models", "")
        except Exception as e:
            if attempt % 10 == 0:
                print(f"[vllm-server] Попытка {attempt+1}/{max_retries}... ({str(e)[:50]})")
            time.sleep(1)
    
    print("[vllm-server] ✗ Сервер не запустился за отведенное время")
    process.terminate()
    raise RuntimeError("vLLM server initialization timeout")


def check_vllm_health(api_base: str) -> bool:
    """Проверяет здоровье vLLM сервера."""
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(
                f"{api_base}/models",
                method="GET"
            ),
            timeout=2
        )
        return response.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    # Пример запуска
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    process, api_base = launch_vllm_server(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path="/path/to/adapter",
        gpu_memory_fraction=0.85,
    )
    
    print(f"[vllm-server] API доступен по адресу: {api_base}")
    print("[vllm-server] Нажмите Ctrl+C для остановки")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("[vllm-server] Остановка...")
        process.terminate()
        process.wait()
