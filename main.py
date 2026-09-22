import os, json, time, re
from typing import Any, Dict, List
from llama_cpp import Llama
from rich.console import Console

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA
from gbnf import agent_grammar

console = Console()
tools_json = json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2)

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
    model_path = "./Qwen2.5-7B-Instruct-Q5_K_M.gguf"
    console.print(f"[dim]used model: {model_path}[/dim]")
    cpu_cores = max(1, (os.cpu_count() or 4) - 2)
    console.print(f"[dim]detected max CPU cores: {cpu_cores}[/dim]")
    console.print(f"[dim]loading model...[/dim]")

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=cpu_cores,
        n_gpu_layers=0,
        chat_format="chatml",
        verbose=False
    )

    system_prompt = (
        "당신은 유능하고 친절한 한국어 AI 비서입니다.\n"
        "사용자의 입력에 대한 답변을 하는것이 아닌 의도를 파악하세요.\n"
        f"사용 가능한 도구 목록({tools_json})을 참고하여 필요 도구를 판단합니다.\n"
        "도구 실행시 아래 포맷을 포함하세요:\n"
        "<tool_call>\n"
        '{"name": "도구이름", "arguments": {"인자": "값"}}\n'
        "</tool_call>\n"
    )
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

        tool_call = parse_tool_call(assistant_text)

        # Case 1: 도구 실행
        if tool_call:
            fn_name = tool_call.get("name", "")
            fn_args = tool_call.get("arguments", {})

            console.print(f"[dim][도구 실행 요청 감지: {fn_name} | 인자: {fn_args}][/dim]")

            if fn_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
            else:
                result = f"알 수 없는 도구: {fn_name}"

            messages.append({"role": "assistant", "content": assistant_text})

            tool_msg = (
                f"<tool_response>\n"
                f'{{"name": "{fn_name}", "result": {json.dumps(result, ensure_ascii=False)}}}\n'
                f"</tool_response>\n"
            )
            messages.append({"role": "user", "content": tool_msg})

            console.print(f"[dim]generating final response...[/dim]")

            final_res = llm.create_chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=512,
                stop=["<|im_end|>"]
            )
            final_text = final_res["choices"][0]["message"]["content"] or ""

            console.print(f"[bold blue]Agent:[/bold blue] {final_text}")
            messages.append({"role": "assistant", "content": final_text})
        # Case 2: 일반 대화
        else:
            console.print(f"[bold blue]Agent:[/bold blue] {assistant_text}")
            messages.append({"role": "assistant", "content": assistant_text})

        console.print(f"[dim]({time.time() - t_start:.2f}s)[/dim]")

if __name__ == "__main__":
    main()