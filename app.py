import streamlit as st
from openai import OpenAI
from tools.prompt import generate_prompt
from tools.rag import RAG


@st.cache_resource
def initialize_components():
    # 1. 初始化 RAG
    rag = RAG(file_path="docs/软件测试知识库.txt", k=3)

    # 2. 初始化 DeepSeek 客户端
    def get_deepseek_client():
        return OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )

    client = get_deepseek_client()
    return rag, client


rag, client = initialize_components()


# 封装对话逻辑,流式输出
def generate_response_stream(client, enhanced_prompt):
    stream = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "你是软件测试专家，结合以下上下文回答问题"},
            {"role": "user", "content": enhanced_prompt}
        ],
        stream=True  # 启用流式传输
    )
    reasoning_content = ""
    full_content = ""
    # 处理流式响应
    try:

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"发生错误：{str(e)}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_content
    })
    # print(reasoning_content)
    # for chunk in stream:
    #     if chunk.choices[0].delta.reasoning_content:
    #         reasoning_content += chunk.choices[0].delta.reasoning_content
    #         # yield chunk.choices[0].delta.reasoning_content
    #     elif chunk.choices[0].delta.content:
    #         full_content += chunk.choices[0].delta.content
    #         yield chunk.choices[0].delta.content

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
    enhanced_prompt = generate_prompt(prompt, context, st.secrets["DEEPSEEK_API_KEY"])
    with st.spinner("正在生成回答..."):
        try:
            st.chat_message("assistant").write_stream(
                generate_response_stream(client, enhanced_prompt)
            )
        except Exception as e:
            answer = f"发生错误：{str(e)}"
    # st.session_state.messages.append({"role": "assistant", "content": answer})
    # st.chat_message("assistant").write(answer)

if st.button("清除会话历史"):
    st.session_state.messages = [{"role": "system", "content": "你是软件测试智能助教"}]
    st.rerun()
