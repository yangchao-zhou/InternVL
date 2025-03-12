from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
path = '/mnt/workspace/yangchao.zhou/opt/models/OpenGVLab/InternVL2-8B'
model = path
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
