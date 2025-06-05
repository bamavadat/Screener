# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-0aec65f2f73b4afab86063184d94cf8f", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

# Print the response
print("Response:", response.choices[0].message.content)
print("\n" + "=" * 50)

# Print usage statistics
if hasattr(response, 'usage') and response.usage:
    usage = response.usage
    print(f"Token Usage:")
    print(f"  Prompt tokens: {usage.prompt_tokens}")
    print(f"  Completion tokens: {usage.completion_tokens}")
    print(f"  Total tokens: {usage.total_tokens}")

    # DeepSeek pricing (as of 2024 - verify current rates)
    # DeepSeek-Chat: $0.14 per 1M input tokens, $0.28 per 1M output tokens
    input_cost_per_1m = 0.14
    output_cost_per_1m = 0.28

    input_cost = (usage.prompt_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (usage.completion_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    print(f"\nCost Breakdown:")
    print(f"  Input cost: ${input_cost:.6f}")
    print(f"  Output cost: ${output_cost:.6f}")
    print(f"  Total cost: ${total_cost:.6f}")
else:
    print("Usage information not available in response")
