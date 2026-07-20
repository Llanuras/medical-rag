from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2,
)

response = llm.invoke(
    "请用中文简要说明RAG在医学专业知识问答中的作用，控制在150字以内。"
)

print(response.content)
