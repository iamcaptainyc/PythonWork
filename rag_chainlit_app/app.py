# app.py

import chainlit as cl
from rag_chain import create_conversational_qa_chain, invoke_qa_chain

qa_chain = None

@cl.on_chat_start
async def start():
    files = await cl.AskFileMessage(
        content="📄 请上传一个 PDF 文件以开始问答",
        accept=["application/pdf"],
        max_size_mb=20,
        max_files=1,
    ).send()

    file = files[0]
    pdf_path = f"./data/{file.name}"
    if file.path:
        # 你可以选择复制一份保存到你想要的位置
        import shutil
        shutil.copyfile(file.path, pdf_path)
        await cl.Message(content=f"文件已保存到：{pdf_path}").send()
    else:
        await cl.Message(content="文件路径无效").send()

    global qa_chain
    qa_chain = create_conversational_qa_chain(pdf_path)
    cl.user_session.set("chat_history", [])

    await cl.Message(content="✅ 文档已加载，现在你可以提问啦！").send()

@cl.on_message
async def chat(message: cl.Message):
    history = cl.user_session.get("chat_history", [])
    res, new_history, _ = await cl.make_async(invoke_qa_chain)(qa_chain, message.content, history)
    cl.user_session.set("chat_history", new_history)
    await cl.Message(content=res).send()
