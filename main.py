import os, json, time, re
from typing import Any, Dict, List
from llama_cpp import Llama
from rich.console import Console

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA

console = Console()
PROFILE_PATH = "user_profile.json"

def load_profile() -> Dict[str, str]:
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    else:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    return {}

def save_profile(profile: Dict[str, str]) -> None:
    try:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    except Exception as e:
        console.print(f"[bold red]프로필 저장 실패: {e}[/bold red]")

def build_system_prompt(user_profile: Dict[str, str] | None = None) -> str:
    tools_json = json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2)

    profile_section = ""
    if user_profile:
        profile_lines = "\n".join(f"- {k}: {v}" for k, v in user_profile.items())
        profile_section = f"\n[사용자에 대해 알고 있는 정보]\n{profile_lines}\n"

    return (
        "당신은 유능하고 친절한 한국어 AI 비서입니다.\n"
        "일상 대화나 단순 질문에는 도구를 쓰지 말고 자연스럽게 한국어로 대화하세요.\n"
        
        "사용자와의 관계를 형성하는 대화는 중요하게 생각하세요.\n"
        f"{profile_section}"
        "사용자의 메시지에서 이름, 직업, 취향, 습관처럼 장기적으로 기억해둘 만한 개인 정보가 "
        "새로 나오면, 평소처럼 답변한 뒤 맨 끝에 다음 형식으로 덧붙이세요 (없으면 절대 붙이지 마세요):\n"
        "<memory>\n"
        '{"항목명": "내용"}\n'
        "</memory>\n"
        "이 태그는 화면에 보이지 않으니 자연스럽게 답변 뒤에 붙이면 됩니다.\n\n"
        
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

def extract_and_strip_memory_tag(text: str) -> tuple[str, Dict[str, str]]:
    match = re.search(r"<memory>\s*({.*?})\s*(?:</memory>)?", text, re.DOTALL)

    raw_json = match.group(1).strip()
    clean_text = (text[:match.start()] + text[match.end():]).strip()

    # 마크다운 백틱(```json ... ```) 제거
    raw_json = re.sub(r"^```(?:json)?|```$", "", raw_json, flags=re.MULTILINE).strip()

    try:
        facts = json.loads(raw_json)
    except json.JSONDecodeError:
        try:
            fixed_json = raw_json.replace("'", '"')
            facts = json.loads(fixed_json)
        except Exception:
            return clean_text, {}

    if isinstance(facts, dict):
        valid_facts = {
            str(k): str(v) for k, v in facts.items()
            if isinstance(v, (str, int, float)) and str(v).strip()
        }
        return clean_text, valid_facts

    return clean_text, {}

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
    
    user_profile = load_profile()
    if user_profile:
        console.print(f"[dim]불러온 사용자 프로필: {user_profile}[/dim]")

    system_prompt = build_system_prompt(user_profile)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # intial dialogue
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.1,
        max_tokens=100,
        stop=["</tool_call>", "<|im_end|>"]
    )
    
    assistant_text = response["choices"][0]["message"]["content"] or ""
    assistant_text, new_facts = extract_and_strip_memory_tag(assistant_text)
    
    if new_facts:
        user_profile.update(new_facts)
        save_profile(user_profile)
        messages[0]["content"] = build_system_prompt(user_profile)
        console.print(f"[dim]💾 기억함: {new_facts}[/dim]")

    console.print(f"[bold blue]Agent:[/bold blue] {assistant_text}")
    messages.append({"role": "assistant", "content": assistant_text})
            
    while True:
        user_input = console.input("[bold green]You:[/bold green] ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        
        console.print(f"[dim]thinking...[/dim]")
        t_start = time.time()
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=512,
            stop=["</tool_call>", "<|im_end|>"]
        )

        assistant_text = response["choices"][0]["message"]["content"] or ""
        console.print(f"[dim]{assistant_text}[/dim]")
        
        if "<tool_call>" in assistant_text and "</tool_call>" not in assistant_text:
            assistant_text += "\n</tool_call>"

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
            
            final_text, new_facts = extract_and_strip_memory_tag(final_text)
            if new_facts:
                user_profile.update(new_facts)
                save_profile(user_profile)
                messages[0]["content"] = build_system_prompt(user_profile)
                console.print(f"[dim]💾 기억함: {new_facts}[/dim]")

            console.print(f"[bold blue]Agent:[/bold blue] {final_text}")
            messages.append({"role": "assistant", "content": final_text})

        # Case 2: 일반 대화
        else:
            clean_text, new_facts = extract_and_strip_memory_tag(assistant_text)
            
            if new_facts:
                user_profile.update(new_facts)
                save_profile(user_profile)
                messages[0]["content"] = build_system_prompt(user_profile)
                console.print(f"[dim]💾 기억함: {new_facts}[/dim]")

            console.print(f"[bold blue]Agent:[/bold blue] {clean_text}")
            messages.append({"role": "assistant", "content": clean_text})

        console.print(f"[dim]({time.time() - t_start:.2f}s)[/dim]")

if __name__ == "__main__":
    main()