import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

model = ChatOllama(model="llama3.2:3b", temperature=0.1)

@tool
def read_file_content(file_path: str) -> str:
    """특정 파일의 내용을 읽어서 반환합니다."""
    if not os.path.exists(file_path):
        return f"오류: '{file_path}' 파일이 존재하지 않습니다."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"파일 읽기 실패: {str(e)}"

@tool
def write_file_content(file_path: str, content: str) -> str:
    """지정한 파일에 내용을 작성(저장)합니다."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"성공적으로 '{file_path}' 파일에 내용을 저장했습니다."
    except Exception as e:
        return f"파일 저장 실패: {str(e)}"

@tool
def list_directory_files(directory_path: str = ".") -> str:
    """지정한 디렉토리 내부의 파일 목록을 조회합니다."""
    try:
        files = os.listdir(directory_path)
        return "\n".join(files) if files else "디렉토리가 비어 있습니다."
    except Exception as e:
        return f"디렉토리 조회 실패: {str(e)}"

tools = [read_file_content, write_file_content, list_directory_files]

system_prompt = """
당신은 사용자의 컴퓨터 파일 시스템에 접근하여 요청된 작업을 수행하는 유능한 AI 에이전트입니다.
한국어로 답변하세요.
"""

agent_executor = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

while True:
    user_input = input("\nuser: ")
    if user_input.lower() == 'q':
        break

    inputs = {"messages": [("user", user_input)]}
    response = agent_executor.invoke(inputs)

    final_message = response["messages"][-1]
    print(f"\nAI: {final_message.content}")