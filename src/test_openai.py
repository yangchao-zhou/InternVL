from openai import OpenAI
'''
启动server 服务

lmdeploy serve api_server /mnt/workspace/yangchao.zhou/opt/models/OpenGVLab/Mini-InternVL-Chat-2B-V1-5 --backend turbomind --server-port 23334
Mini-InternVL-Chat-2B-V1-5
InternVL2-8B 
'''
client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)