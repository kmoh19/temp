from openai import OpenAI

endpoint = "https://guardrail-2026.services.ai.azure.com/openai/v1/"
model_name = "Kimi-K2.5"
deployment_name = "Kimi-K2.5"

api_key = "<your-api-key>"

client = OpenAI(
    base_url=f"{endpoint}",
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)
