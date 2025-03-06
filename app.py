import streamlit as st
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