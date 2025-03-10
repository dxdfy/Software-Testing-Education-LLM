from openai import OpenAI
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