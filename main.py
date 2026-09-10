import time
from rich.console import Console
from dataclasses import dataclass, field
from typing import Any
from llama_cpp import Llama

@dataclass
class Agent:
    model_path: str = "./qwen2.5-coder-3b-instruct.gguf"
    system_prompt: str = "당신은 유능한 AI 비서입니다. 한국어로 대화합니다."
    messages: list[dict[str, Any]] = field(default_factory=list)
    llm: Llama = field(init=False, repr=False)
    
    def __post_init__(self) -> None:
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            verbose=False,
            n_gpu_layers=0  # CPU only
        )
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        response = self.llm.create_chat_completion(
            messages=self.messages,
            temperature=0.3,
        )
        
        choices = response.get("choices")
        if not choices:
            raise RuntimeError("Model response missing choices")
        
        message = choices[0].get("message")
        if message is None:
            raise RuntimeError("Model response missing message")
        
        answer = message.get("content") or ""
        self.messages.append({"role": "assistant", "content": answer})
        return answer

def main() -> None:
    agent = Agent(model_path="./qwen2.5-coder-3b-instruct.gguf")
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