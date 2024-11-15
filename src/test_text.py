import base64
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://8.130.50.247:23334/v1')
model_name = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image',
        }]
    }],
    temperature=0.8,
    top_p=0.8
)
print(response)
