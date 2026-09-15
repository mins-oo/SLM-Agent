import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List
from llama_cpp import Llama
from rich.console import Console

console = Console()

# =========================================================
# 1. 샌드박스 경로 안전 검증 & 실행 승인
# =========================================================
SANDBOX_ROOT = Path(os.getcwd()).resolve()

def resolve_safe_path(path_str: str) -> Path:
    target = (SANDBOX_ROOT / path_str).resolve()
    if SANDBOX_ROOT not in target.parents and target != SANDBOX_ROOT:
        raise ValueError(f"샌드박스 허용 경로 밖으로의 접근은 불가합니다: {path_str}")
    return target

def ask_approval(tool_name: str, args: dict) -> bool:
    console.print(f"[bold yellow]⚠ 승인 요청[/bold yellow] | 도구: [bold]{tool_name}[/bold] | 인자: {args}")
    try:
        ans = console.input("[bold yellow]실행하시겠습니까? (y/N): [/bold yellow]").strip().lower()
        return ans in {"y", "yes"}
    except (KeyboardInterrupt, EOFError):
        return False

# =========================================================
# 2. 파일 작업 도구(Tool) 파이썬 기능 구현
# =========================================================
def tool_read_file(path: str) -> str:
    if not ask_approval("read_file", {"path": path}):
        return "사용자가 파일 읽기 실행을 거부했습니다."
    try:
        p = resolve_safe_path(path)
        if not p.is_file():
            return f"파일이 존재하지 않습니다: {path}"
        content = p.read_text(encoding="utf-8")
        return content[:4000] + ("\n...(생략)" if len(content) > 4000 else "")
    except Exception as e:
        return f"파일 읽기 실패: {e}"

def tool_write_file(path: str, content: str = "") -> str:
    if not ask_approval("write_file", {"path": path, "content_len": len(content)}):
        return "사용자가 파일 쓰기 실행을 거부했습니다."
    try:
        p = resolve_safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"성공적으로 파일이 작성되었습니다: {path}"
    except Exception as e:
        return f"파일 쓰기 실패: {e}"

def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    if not ask_approval("edit_file", {"path": path, "old_text": old_text, "new_text": new_text}):
        return "사용자가 파일 수정 실행을 거부했습니다."
    try:
        p = resolve_safe_path(path)
        if not p.is_file():
            return f"파일이 존재하지 않습니다: {path}"
        text = p.read_text(encoding="utf-8")
        if old_text not in text:
            return f"파일 내에서 '{old_text}' 텍스트를 찾을 수 없습니다."
        updated = text.replace(old_text, new_text, 1)
        p.write_text(updated, encoding="utf-8")
        return f"성공적으로 파일을 수정했습니다: {path}"
    except Exception as e:
        return f"파일 수정 실패: {e}"

TOOL_FUNCTIONS = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}

# =========================================================
# 3. Qwen2.5 시스템 프롬프트 및 도구 정의
# =========================================================
TOOLS_SCHEMA = [
    {
        "name": "read_file",
        "description": "지정된 경로의 텍스트 파일 내용을 읽어옵니다.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "읽을 파일 경로"}},
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "지정된 경로에 새 파일을 생성하거나 내용을 덮어씁니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "작성할 파일 경로"},
                "content": {"type": "string", "description": "작성할 텍스트 내용"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "파일 내 기존 텍스트(old_text)를 새 텍스트(new_text)로 1회 치환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "수정할 파일 경로"},
                "old_text": {"type": "string", "description": "바꿀 기존 텍스트"},
                "new_text": {"type": "string", "description": "새로 들어갈 텍스트"}
            },
            "required": ["path", "old_text", "new_text"]
        }
    }
]

def build_system_prompt() -> str:
    tools_json = json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2)
    return (
        "당신은 유능하고 친절한 한국어 AI 비서입니다.\n"
        "일상 대화나 단순 질문에는 도구를 쓰지 말고 자연스럽게 한국어로 대화하세요.\n\n"
        "파일 읽기, 생성, 수정 등 실제 작업이 필요할 때만 다른 텍스트 없이 아래 포맷으로만 응답하세요:\n"
        "<tool_call>\n"
        '{"name": "도구이름", "arguments": {"인자": "값"}}\n'
        "</tool_call>\n\n"
        f"[사용 가능한 도구 목록]\n{tools_json}"
    )

def parse_tool_call(text: str) -> dict | None:
    text = text.strip()
    
    # <tool_call> 태그 내부 추출
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

# =========================================================
# 4. 메인 대화 루프
# =========================================================
def main():
    model_path = "./Qwen2.5-7B-Instruct-Q5_K_M.gguf"

    if not os.path.exists(model_path):
        console.print(f"[bold red]오류:[/bold red] Qwen2.5 GGUF 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)

    # CPU 자원 최적화 할당
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
            # 1차 추론 (stop 토큰으로 도구 호출 시점 중단)
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                stop=["</tool_call>", "<|im_end|>"]
            )

            assistant_text = response["choices"][0]["message"]["content"] or ""
            
            # 태그 보완
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

                # 도구 결과를 바탕으로 최종 한국어 응답 생성
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