# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from utils import retrieve_docs, retrieve_longterm, call_vllm_chat, save_to_longterm
import uvicorn
import os
import threading
import ingest  # optional: reuse ingest.py

app = FastAPI(title="RAG with vLLM - Demo")

# 简单短期会话存储（生产环境用 Redis/DB）
# user_id -> list of {"role": "user|assistant", "content": "..."}
SHORT_TERM_CONTEXT: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_TURNS = 6  # 保留最近 6 轮（可调整）

SYSTEM_PROMPT = (
    "你是公司内部的知识助理。回答要基于检索到的文档和用户上下文，"
    "如果检索到的信息与用户问题冲突，优先说明不确定性并引用来源。"
)

class ChatRequest(BaseModel):
    user_id: str
    message: str
    save_memory: Optional[bool] = False  # 是否强制把该条存长期记忆

class ChatResponse(BaseModel):
    reply: str
    retrieved_docs: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_id = req.user_id
    content = req.message.strip()
    if not content:
        raise HTTPException(status_code=400, detail="empty message")

    # 1) 检索长期记忆（如果有）与知识库文档
    longterm_hits = retrieve_longterm(content, k=3)
    docs = retrieve_docs(content, k=4)

    # 2) 构造 messages（system + retrieved docs + short history + user）
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # 把检索到的重要长期记忆注入提示（作为辅助事实）
    if longterm_hits:
        lt_text = "\n\n".join([f"- {d['text']}" for d in longterm_hits])
        messages.append({"role": "system", "content": f"长期记忆检索到如下相关信息：\n{lt_text}"})

    # 把检索到的文档片段注入（带来源 meta）
    if docs:
        doc_text = ""
        for d in docs:
            src = d.get("meta", {}).get("source", "unknown")
            doc_text += f"来源：{src}\n{d['text']}\n---\n"
        messages.append({"role": "system", "content": f"检索到以下知识库片段，回答时可以引用：\n{doc_text}"})

    # 加入短期历史（最近几轮）
    history = SHORT_TERM_CONTEXT.get(user_id, [])
    # 只保留最近的几条（人机对话轮）
    if history:
        # 转换为 messages
        for turn in history[-(MAX_HISTORY_TURNS*2):]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    # 最后一条是用户当前问题
    messages.append({"role": "user", "content": content})

    # 3) 调用 vLLM
    try:
        reply = call_vllm_chat(system_prompt=SYSTEM_PROMPT, messages=messages, temperature=0.2, max_tokens=512)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM call failed: {e}")

    # 4) 更新短期内存（保存本轮对话）
    SHORT_TERM_CONTEXT.setdefault(user_id, []).append({"role": "user", "content": content})
    SHORT_TERM_CONTEXT.setdefault(user_id, []).append({"role": "assistant", "content": reply})
    # 限制长度
    if len(SHORT_TERM_CONTEXT[user_id]) > (MAX_HISTORY_TURNS * 2 + 2):
        SHORT_TERM_CONTEXT[user_id] = SHORT_TERM_CONTEXT[user_id][- (MAX_HISTORY_TURNS * 2):]

    # 5) 如果用户要求或系统策略判定为重要信息则保存到长期记忆
    if req.save_memory or any(k in content for k in ["记住", "我的名字", "我喜欢", "我在"]):
        save_to_longterm(content, meta={"user_id": user_id})
        # 这里仅演示把用户显式要求保存的文本入库，生产应做信息抽取与结构化

    return ChatResponse(reply=reply, retrieved_docs=[{"text": d["text"], "meta": d.get("meta", {})} for d in docs])

@app.post("/ingest")
def run_ingest():
    # 异步调用 ingest（避免阻塞）
    t = threading.Thread(target=ingest.main, daemon=True)
    t.start()
    return {"status": "ingest started"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 9000)), reload=False)
