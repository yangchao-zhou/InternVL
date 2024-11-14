pip install openai

path = '/mnt/workspace/yangchao.zhou/opt/models/OpenGVLab/InternVL2-8B'

export  PYTHONPATH=`pwd`

source /opt/venv/bin/activate

pip install flash-attn==2.3.6 --no-build-isolation

lmdeploy serve api_server /mnt/workspace/yangchao.zhou/opt/models/OpenGVLab/InternVL2-8B --backend turbomind --server-port 23333

vllm serve "/mnt/workspace/yangchao.zhou/opt/models/OpenGVLab/InternVL2-8B" --trust-remote-code --dtype half 


https://internvl.readthedocs.io/en/latest/get_started/local_chat_demo.html



## Streamlit Demo
### Step 1: Set Variables
export SD_SERVER_PORT=39999
export WEB_SERVER_PORT=10003
export CONTROLLER_PORT=40000

export CONTROLLER_URL=http://0.0.0.0:$CONTROLLER_PORT
export SD_WORKER_URL=http://0.0.0.0:$SD_SERVER_PORT

### Step 2: Start the Streamlit Web Server
cd streamlit_demo/
streamlit run app.py --server.port $WEB_SERVER_PORT -- --controller_url $CONTROLLER_URL --sd_worker_url $SD_WORKER_URL

### Step 3: Start the Controller
export CONTROLLER_PORT=40000
cd streamlit_demo/
python controller.py --host 0.0.0.0 --port $CONTROLLER_PORT

### Step 4: Start the Model Workers
export SD_SERVER_PORT=39999
export WEB_SERVER_PORT=10003
export CONTROLLER_PORT=40000

export CONTROLLER_URL=http://0.0.0.0:$CONTROLLER_PORT
export SD_WORKER_URL=http://0.0.0.0:$SD_SERVER_PORT
cd streamlit_demo/
CUDA_VISIBLE_DEVICES=1 python model_worker.py --host 0.0.0.0 --controller $CONTROLLER_URL --port 40004 --worker http://0.0.0.0:40004 --model-path /mnt/workspace/yangchao.zhou/opt/models/OpenGVLab/InternVL2-8B

