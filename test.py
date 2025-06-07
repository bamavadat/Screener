from openai import OpenAI

client = OpenAI(
    base_url="https://api.targon.com/v1",
    api_key="sn4_eoypi1xblk4smc91e8w3sn8yh7jn"
)

try:
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        stream=True,
        messages=[
            {"role": "system", "content": "You are a helpful programming assistant."},
            {"role": "user", "content": "Write a bubble sort implementation in Python with comments explaining how it works"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
except Exception as e:
    print(f"Error: {e}")

    cpk_9be9238cb74f42f286d481a810eeae05.e01f3cd9bdc35f80a9ad91abacd78cc5.zg9UXXctkryoAvBpkXWkcvfISbzWzfZP
