import os, sys, json, time, re
from pathlib import Path
from typing import Any, Dict, List
from llama_cpp import Llama
from rich.console import Console

from tools import SANDBOX_ROOT, TOOL_FUNCTIONS, TOOLS_SCHEMA

console = Console()

def build_system_prompt() -> str:
    tools_json = json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2)
    return (
        "당신은 유능하고 친절한 한국어 AI 비서입니다.\n"
        "일상 대화나 단순 질문에는 도구를 쓰지 말고 자연스럽게 한국어로 대화하세요.\n"
        "파일 읽기, 생성, 수정 등 실제 작업이 필요할 때만 다른 텍스트 없이 아래 포맷으로만 응답하세요:\n"
        "<tool_call>\n"
        '{"name": "도구이름", "arguments": {"인자": "값"}}\n'
        "</tool_call>\n\n"
        f"[사용 가능한 도구 목록]\n{tools_json}"
    )

def parse_tool_call(text: str) -> dict | None:
    text = text.strip()
    
    match = re.search(r"<tool_call>\s*({.*?})\s*(?:</tool_call>)?", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx : end_idx + 1]
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "name" in data:
                if "arguments" not in data:
                    data["arguments"] = {}
                return data
        except json.JSONDecodeError:
            pass

    return None

def main():
    model_path = "./Qwen2.5-7B-Instruct-Q5_K_M.gguf"

    if not os.path.exists(model_path):
        console.print(f"[bold red]오류:[/bold red] Qwen2.5 GGUF 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    cpu_cores = max(1, (os.cpu_count() or 4) - 2)

    console.print(f"[dim]작업 디렉터리: {SANDBOX_ROOT}[/dim]")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=cpu_cores,
        n_gpu_layers=0,
        chat_format="chatml",
        verbose=False
    )

    system_prompt = build_system_prompt()
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        t_start = time.time()

        try:
            # 1차 추론 (stop 토큰으로 도구 호출 중단)
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                stop=["</tool_call>", "<|im_end|>"]
            )

            assistant_text = response["choices"][0]["message"]["content"] or ""
            
            if "<tool_call>" in assistant_text and "</tool_call>" not in assistant_text:
                assistant_text += "\n</tool_call>"

            tool_call = parse_tool_call(assistant_text)

            # Case 도구 실행
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
                    f"위 실행 결과를 바탕으로 사용자에게 한국어로 답변하세요."
                )
                messages.append({"role": "user", "content": tool_msg})

                final_res = llm.create_chat_completion(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=512,
                    stop=["<|im_end|>"]
                )
                final_text = final_res["choices"][0]["message"]["content"]
                console.print(f"[bold blue]Agent:[/bold blue] {final_text}")
                messages.append({"role": "assistant", "content": final_text})
            # Case 일반 대화
            else:
                console.print(f"[bold blue]Agent:[/bold blue] {assistant_text}")
                messages.append({"role": "assistant", "content": assistant_text})

        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            messages.pop()
            continue

        console.print(f"[dim]({time.time() - t_start:.2f}s)[/dim]\n")

if __name__ == "__main__":
    main()