# 使用示例
from openai import OpenAI

def generate_prompt(user_input,context,api_key="sk-03c2c982629b43c4bf87446553dd5429"):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个分类器，将用户给出的问题进行分类，分类为知识性问题和推理问题俩种，只输出问题的类别"},
            {"role": "user", "content": "问题为"+user_input},
        ],
        stream=False
    )
    cla = response.choices[0].message.content
    # print(cla)
    if cla == "知识性问题":
        enhanced_prompt = f"""
                用户问题：{user_input}
                相关上下文：{context}
                要求：根据上下文知识回答用户问题
                """
        return enhanced_prompt
    else:
        enhanced_prompt = f"""
                        用户问题：{user_input}
                        相关上下文：{context}
                        要求：参考上下文知识，将问题分为子问题，一步步推理，严格区分软件缺陷、故障、失效
                        """
        return enhanced_prompt
