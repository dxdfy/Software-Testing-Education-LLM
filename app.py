import streamlit as st
from openai import OpenAI
# 初始化 DeepSeek 客户端
def get_deepseek_client():
    return OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],  # 从 Streamlit secrets 获取 API Key
        base_url="https://api.deepseek.com"
    )

# 封装对话逻辑
def get_deepseek_response(messages):
    client = get_deepseek_client()
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
st.title("软件测试智能助教")
st.caption("A Streamlit chatbot powered by Deepseek")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "我是软件测试智能助教，欢迎向我提问"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = "大模型的回答"
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
if st.button("清除会话历史"):
    st.session_state.messages = [{"role": "assistant", "content": "我是软件测试智能助教，欢迎向我提问"}]