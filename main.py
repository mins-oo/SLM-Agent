import requests
import time
from rich.console import Console
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Agent:
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = field(default="NO_API_KEY", repr=False)
    system_prompt: str = "당신은 유능한 AI 비서입니다. 한국어로 대화합니다. 반드시 모르는 사실에 대해서는 정보가 없다고 답변합니다."
    messages: list[dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        r = requests.post(
            url,
            headers=headers,
            json={
                "model": self.model,
                "messages": self.messages,
            },
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices")
        
        if not choices:
            raise RuntimeError("Model response missing choices")
        
        message = choices[0].get("message")
        if message is None:
            raise RuntimeError("Model response missing message")
        
        response = message.get("content") or ""
        self.messages.append({"role": "assistant", "content": response})
        return response

def main() -> None:
    agent = Agent(model="qwen2.5-coder:7b")
    console = Console()
    
    while True:
        console.print("[bold green]You:[/bold green] ", end="")
        user_input = console.input()
        
        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("[dim]Exiting...[/dim]")
            break
        
        t = time.time()
        
        with console.status("[dim]Thinking...[/dim]", spinner="arc"):
            response = agent.chat(user_input).strip()

        console.print(f"[bold blue]Agent ({time.time() - t:.2f}s):[/bold blue] {response}")

if __name__ == "__main__":
    main()