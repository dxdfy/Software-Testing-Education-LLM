from openai import OpenAI
def get_deepseek_response(messages):
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "你是软件测试专家，结合以下上下文回答问题"},
                {"role": "user", "content": messages}
            ],
        )
        reasoning_content = response.choices[0].message.reasoning_content
        answer = response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    client = OpenAI(api_key="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "给出软件测试的定义"},
        ],
        stream=False
    )
    print(response.choices[0].message.content)