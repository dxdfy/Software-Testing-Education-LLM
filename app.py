import streamlit as st
from openai import OpenAI
from tools.rag import RAG
@st.cache_resource
def initialize_components():
    # 1. 初始化 RAG
    rag = RAG(file_path="docs/ch1软件测试知识库.txt")

    # 2. 初始化 DeepSeek 客户端
    def get_deepseek_client():
        return OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )

    client = get_deepseek_client()

    return rag, client
rag, client = initialize_components()
# 封装对话逻辑
def get_deepseek_response(messages):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


st.title("软件测试智能助教")
st.caption("A Streamlit chatbot powered by Deepseek")
# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "你是软件测试智能助教"}
    ]
# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
# 处理用户输入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner("正在检索知识库..."):
        context = rag.retrieve(prompt)
    with st.expander("查看知识库检索结果", expanded=True):
        st.markdown("#### 相关上下文片段：")
        for i, doc in enumerate(context.split('\n\n'), 1):
            st.markdown(f"**片段 {i}**")
            st.code(doc, language="text")
    enhanced_prompt = f"""
        用户问题：{prompt}
        相关上下文：{context}
        要求：结合上下文给出专业、详细的回答，避免提及知识库来源
        """
    with st.spinner("正在生成回答..."):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是软件测试专家，结合以下上下文回答问题"},
                    {"role": "user", "content": enhanced_prompt}
                ],
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"发生错误：{str(e)}"
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

if st.button("清除会话历史"):
    st.session_state.messages = [{"role": "system", "content": "你是软件测试智能助教"}]
    st.rerun()