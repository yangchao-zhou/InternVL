# 环境二选一
# conda activate infer_tyb
# conda activate rancloud

import os
import pandas as pd
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# from loguru import logger
import html
import jinja2
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_prompt_template(template_file):
    with open(template_file, 'r') as f:
        template = f.read()
    template = jinja2.Template(template)
    return template



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = "/mnt/data/ran.xiao/cloud/prepare_for_online/llama2_as_en_12b_mistral_v4_1021"
output_file_name = 'data/eval/llama2_as_en_12b_mistral_v4_1021-20241114.xlsx' # 可以，多样性差点





tokenizer = AutoTokenizer.from_pretrained(model_path)
terminators = [
    tokenizer.eos_token_id,
    ]

generate_config = GenerationConfig(**{
    "do_sample": True,
    "max_new_tokens": 256,
    "repetition_penalty": 1.05,
    "length_penalty": 1,
    "temperature": 0.85,
    "top_k": 7,
    "top_p": 0.9,
    "min_len": 2,
    "pad_token_id": tokenizer.eos_token_id,
    "no_repeat_ngram_size": 0,
    "frequency_penalty":0,
    "eos_token_id": terminators
    })


# ## sys_20_2
system_prompt_template_nostruct = load_prompt_template("/mnt/data/ran.xiao/cloud/eval/config_data/nsfw_npc_sys_prompt_0920_nostruct.tmpl")
system_prompt_template_struct = load_prompt_template("/mnt/data/ran.xiao/cloud/eval/config_data/nsfw_npc_sys_prompt_0920_struct.tmpl")


# llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=4096, dtype="bfloat16")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # device_map="auto",
    device_map={"": "cuda:1"}  # 显式指定 GPU 0
)

print(model.device)
switch = False
# def generate(messages, model=False):
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     outputs = llm.generate([prompt,], sampling_params=sampling_params)
#     generated_text_list = [output.outputs[0].text for output in outputs]
#     # global switch
#     # if not switch:
#         # logger.info(f"prompt:{prompt}")
#         # logger.info(f"generated_text: {generated_text_list[0]}")
#         # switch = True
#     return generated_text_list[0]

def generate(messages):        
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
        ).to(model.device)


    # print(tokenizer.decode(input_ids[0], skip_special_tokens=False))

    # generate_config 需自定义
    gen_kwargs = dict(
        inputs=input_ids,
        generation_config=generate_config,
        )   
    outputs = model.generate(**gen_kwargs)
    response = outputs[0][input_ids.shape[-1]:]
    len_token_res = len(response)
    return tokenizer.decode(response, skip_special_tokens=True), len_token_res


def user(input):
    return {'role': "user", "content": input}

def assistant(input):
    return {'role': "assistant", "content": input}


def append_to_data_list(data_list, chat_id, round_id, npc_sid, total_number, role_name, intro, greeting, npc_info,user_info, sender_type,content, task_name):
    data_list.append({
        '专项': task_name,
        'chat_id': chat_id,
        'round_id': round_id,
        'npc_sid': npc_sid,
        'total_number': total_number,
        'role_name': role_name,
        'intro': intro,
        'greeting': greeting,
        'npc_info':npc_info,
        'user_info':user_info,
        'sender_type': sender_type,
        'content': content
    })


def infer():
    # 文件路径
    file_path = "/mnt/data/ran.xiao/cloud/eval/test_data_input/固定评测模版v2.0.xlsx"
    df = pd.read_excel(file_path, sheet_name="固定话术-跑模型以此为准！")
    print(df.columns)
    # 读取"角色池"表格，指定所需的列
    character_pool_df = pd.read_excel(file_path, sheet_name="角色池", usecols=["角色名", "struct_info", "intro", "profile", "测试环境sid","greeting"], dtype=str).fillna("")

    # 读取"固定话术"表格，指定所需的列
    fixed_script_df = pd.read_excel(file_path, sheet_name="固定话术-跑模型以此为准！", usecols=["type", "session_id","round", "角色sid", "content"], dtype=str).fillna("")

    # 将每个字段内容以列表形式返回
    character_pool_df = character_pool_df.applymap(lambda x: html.unescape(x) if isinstance(x, str) else x)
    fixed_script_df = fixed_script_df.applymap(lambda x: html.unescape(x) if isinstance(x, str) else x)


    # 将每个字段内容以列表形式返回
    character_names = character_pool_df["角色名"].tolist()

    struct_info = character_pool_df["struct_info"].tolist()

    profile = character_pool_df["profile"].tolist()

    intro = character_pool_df["intro"].tolist()
    greeting = character_pool_df["greeting"].tolist()

    test_env_sid = character_pool_df["测试环境sid"].tolist()
    round_list = fixed_script_df["round"].tolist()
    type_1 =  fixed_script_df["type"].tolist()
    content_list = fixed_script_df["content"].tolist()

    # 初始化一个空列表用于存储最终写入的数据
    sfw_data = []

    # 运行测试
    chat_id = 1

    for j in range(len(character_names)):
        # 批跑包括没有profile在内的角色
        # if pd.isna(profile[j]):
        #     continue

        test_env_sid_value = test_env_sid[j]
        npc_sid = test_env_sid_value
        role_name = character_names[j]
        intro_text = intro[j]
        greeting_text = greeting[j]
        npc_info =struct_info[j]
        user_info = profile[j]
        # sfw = NEW_SYSTEM_PROMPT.format(npc_name=character_names[j], intro=intro[j], npc_profile=struct_info[j])
        # TODO: system change here 0920!
        if system_prompt_template_struct and npc_info:
            npc_info_evaled = eval(npc_info)
            if 'npc_personality' in npc_info_evaled:
                npc_personality = ['npc_personality']
            else:
                npc_personality = ""
            if 'npc_quirks' in npc_info_evaled:
                npc_quirks = ['npc_quirks']
            else:
                npc_quirks = ""
            sfw = system_prompt_template_struct.render({"npc_name": role_name, "intro": intro_text, "npc_profile": npc_info, "npc_personality": npc_personality, "npc_quirks": npc_quirks})
        else:
            sfw = system_prompt_template_nostruct.render({"npc_name": role_name, "intro": intro_text})

        # round_id = 1
        # message_nsfw = [{"role": "system", "content": nsfw}]
        print('  character_names:',character_names[j])
        for i, round in enumerate(round_list):
            round = int(round)
            task_name = None
            if test_env_sid_value == fixed_script_df["角色sid"][i]:
                scene = "easy talk"
                user_input = content_list[i]
                task_name = "简单输入"


            if task_name:
                # SFW
                # if (("OnlineMistral" in model_path) or (model_path.endswith("ModifiedChatTemplate"))) and round == 1: 
                #     message_sfw = [{"role": "system", "content": sfw + "\n\n" + greeting_text}]
                # elif round == 1: 
                if round == 1: 
                    message_sfw = [{"role": "system", "content": sfw}, {"role": "assistant", "content": greeting_text}]
                message_sfw.append(user(user_input))
                response_sfw, len_token = generate(messages=message_sfw)
                print([user_input, response_sfw, str(len(response_sfw)), str(len_token)])

                append_to_data_list(sfw_data, chat_id, round, npc_sid, i + 1, role_name, intro_text, greeting_text, npc_info,user_info,'user', user_input, type_1[i])
                append_to_data_list(sfw_data, chat_id, round, npc_sid, i + 1, role_name, intro_text, greeting_text, npc_info,user_info,'ai', response_sfw, type_1[i])

                message_sfw.append(assistant(response_sfw))
                # round_id += 1

        chat_id += 1

    # 将结果写入Excel
    sfw_df = pd.DataFrame(sfw_data)

    sfw_df.to_excel(output_file_name, index=False)

    print(f"数据已成功写入 {output_file_name}")


def main():
    infer()


if __name__ == "__main__":
    main()