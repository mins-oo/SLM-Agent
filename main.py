import os, json, time, re
from pathlib import Path
from typing import Any, Dict, List
from llama_cpp import Llama
from rich.console import Console

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA, load_memory
from gbnf import agent_grammar

console = Console()
tools_json = json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2)

MODEL_DIR = "./model"

def get_model() -> str:
    dir_path = Path(MODEL_DIR)
    model_files = sorted(list(dir_path.glob("*.gguf")))
    if not model_files:
        raise FileNotFoundError(
            f"'{MODEL_DIR}' folder doesn't have model. Please download GGUF file.\n"
            "recommend: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF"
        )
    return str(model_files[0])

def build_system_prompt(user_memory: dict | None = None) -> str:
    memory_section = ""
    if user_memory:
        memory_lines = "\n".join(f"- {k}: {v}" for k, v in user_memory.items())
        memory_section = f"\n[user info]\n{memory_lines}\n"

    return (
        "당신은 유능하고 친절한 한국어 AI 비서입니다.\n"
        "사용자의 입력에 대한 답변을 하는것이 아닌 의도를 파악하세요.\n"
        f"사용 가능한 도구 목록({tools_json})을 참고하여 필요 도구를 판단합니다.\n"
        f"{memory_section}"
        "사용자의 이름, 직업, 취향, 습관처럼 앞으로도 계속 기억해두면 좋을 정보가 새로 나오면 "
        "remember 도구를 사용해 저장하세요. 이미 기억하고 있는 것과 같은 내용이면 다시 저장하지 마세요.\n"
        "도구 실행시 아래 포맷을 포함하세요:\n"
        "<tool_call>\n"
        '{"name": "도구이름", "arguments": {"인자": "값"}}\n'
        "</tool_call>\n"
    )

def parse_tool_call(text: str) -> dict | None:
    match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", text.strip(), re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        if isinstance(data, dict) and "name" in data:
            data.setdefault("arguments", {})
            return data
    except json.JSONDecodeError:
        pass
    return None

def main():
    model_path = get_model()
    console.print(f"[dim]model: {model_path}[/dim]")
    cpu_cores = max(1, (os.cpu_count() or 4) - 2)
    console.print(f"[dim]using CPU cores: {cpu_cores}[/dim]")

    user_memory = load_memory()
    if user_memory:
        console.print(f"[dim]loaded user info: {list(user_memory.keys())}[/dim]")
    else:
        console.print("[dim]new user detected. welcome![/dim]")

    console.print(f"[dim]loading model...[/dim]")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=cpu_cores,
        n_gpu_layers=0,
        chat_format="chatml",
        verbose=False
    )
    system_prompt = build_system_prompt(user_memory)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.1,
        max_tokens=1
    )
    console.print("[dim]complete![/dim]")
    
    while True:
        user_input = console.input("[bold green]You:[/bold green] ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user_input})
        
        t_start = time.time()
        console.print(f"[dim]thinking...[/dim]")
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=100,
            stop=["<|im_end|>"],
            grammar=agent_grammar
        )
        assistant_text = response["choices"][0]["message"]["content"] or ""
        console.print(f"[dim]{assistant_text}[/dim]")
        messages.append({"role": "assistant", "content": assistant_text})

        tool_call = parse_tool_call(assistant_text)
        if not tool_call:
            console.print("[dim]error: no tool call detected.[/dim]")
            continue
        fn_name = tool_call.get("name", "")
        fn_args = tool_call.get("arguments", {})

        console.print(f"[dim]detected tool: {fn_name} {fn_args}[/dim]")

        if fn_name in TOOL_FUNCTIONS:
            result = TOOL_FUNCTIONS[fn_name](**fn_args)
        else:
            result = f"error: {fn_name}"

        # update system prompt if memory is changed
        if fn_name in {"remember", "forget"}:
            user_memory = load_memory()
            messages[0]["content"] = build_system_prompt(user_memory)
        
        result_msg = f'tool result: {{"name": "{fn_name}", "result": {json.dumps(result, ensure_ascii=False)}}}'
        messages.append({"role": "user", "content": result_msg})
        console.print(f"[dim]generating response...[/dim]")
        final_res = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stop=["<|im_end|>"]
        )
        final_text = final_res["choices"][0]["message"]["content"] or ""
        console.print(f"[bold blue]Agent:[/bold blue] {final_text}")
        messages.append({"role": "assistant", "content": final_text})
        console.print(f"[dim]({time.time() - t_start:.2f}s)[/dim]")

if __name__ == "__main__":
    main()