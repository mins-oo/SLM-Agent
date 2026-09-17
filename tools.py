import os
from pathlib import Path
from rich.console import Console

console = Console()

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

# Tool
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