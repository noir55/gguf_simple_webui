#!/bin/env python3

import argparse
import gradio as gr
import time
import numpy as np
import os
import re


#------
# 設定
#------

# バージョン
VERSION = "1.0.0"

# ページの最上部に表示させたいタイトルを設定
TITLE_STRINGS = "GGUF Chat"

# モデルタイプ("llama")
MODEL_TYPE = "llama"
# ベースモデルファイルを指定
BASE_MODEL = "/path/to/model_file.gguf"

# LoRAのパス(空文字列に設定すると読み込まない)
LORA_WEIGHTS = ""

# コンテキストサイズの指定
CONTEXT_SIZE = 4096
# バッチサイズの指定
BATCH_SIZE = 512

# GPUを使用するかどうか
USE_GPU = "off"
# GPUで処理するレイヤ数
GPU_LAYERS = 40

# 複数GPUで分割処理する割合指定
TENSOR_SPLIT = None

# CPUで処理する場合の、使用CPUコア数(0:自動で決定)
THREAD_NUM = 0

# MMAPが使用できる場合に使用するか
USE_MMAP = "off"
# MLOCKを使用するか
USE_MLOCK = "on"

# プロンプトタイプ("rinna","vicuna","alpaca","llama2","openbuddy","airoboros","beluga","ja-stablelm","mixtral","swallow","nekomata","elyzallama2","karakuri","gemma", "chatml","qa","none")
PROMPT_TYPE = "rinna"
# プロンプトが何トークンを超えたら履歴を削除するか
PROMPT_THRESHOLD = 4096
# 履歴を削除する場合、何トークン未満まで削除するか
PROMPT_DELETED = 2048

# 繰り返しペナルティ(大きいほど同じ繰り返しを生成しにくくなる)
REPETITION_PENALTY = 1.1
# 推論時に生成する最大トークン数
MAX_NEW_TOKENS = 1024
# 推論時の出力の多様さ(大きいほどバリエーションが多様になる)
TEMPERATURE = 0.7

# Rope Scalingに関する設定
ROPE_BASE=10000.0
ROPE_SCALE=1.0

# WebUIがバインドするIPアドレス
GRADIO_HOST = '127.0.0.1'
# WebUIがバインドするポート番号
GRADIO_PORT = 7860

# WebUI上に詳細設定を表示するか
SETTING_VISIBLE = "on"

# デバッグメッセージを標準出力に表示するか("on","off")
DEBUG_FLAG = "on"


#------------------
# クラス、関数定義
#------------------

def user(message, history):
    if history == None:
        history = []
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

# Regenerateボタンクリック時の動作
def regen(history):
    if len(history) == 0:
        return "", [["", ""]]
    else:
        history[-1][1]=""
        return history[-1][0], history

# Remove lastボタンクリック時の動作
def remove_last(history):
    if len(history) == 0:
        return "", [["", ""]]
    else:
        history.pop(-1)
        return history

# プロンプト文字列を生成する関数
def prompt(curr_system_message, history):

    # Vicuna形式のプロンプト生成
    if PROMPT_TYPE == "vicuna":
        prefix = f"""A chat between a curious user and an artificial intelligence assistant.{new_line}The assistant gives helpful, detailed, and polite answers to the user's questions.{new_line}{new_line}"""
        messages = curr_system_message + \
            new_line.join([new_line.join(["USER: "+item[0], "ASSISTANT: "+item[1]])
                    for item in history])
        messages = prefix + messages
    # Alpaca形式のプロンプト生成
    elif PROMPT_TYPE == "alpaca":
        prefix = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.{new_line}{new_line}"""
        messages = curr_system_message + \
            f"{new_line}{new_line}".join([new_line.join([f"### Instruction:{new_line}"+item[0], f"{new_line}### Response:{new_line}"+item[1]])
                    for item in history])
        messages = prefix + messages
    # Llama2 Chat形式のプロンプト生成
    elif PROMPT_TYPE == "llama2":
        prefix = f"""System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.{new_line}"""
        messages = curr_system_message + \
            new_line.join([new_line.join([f"User: "+item[0], f"Assistant: "+item[1]])
                    for item in history])
        messages = prefix + messages
    # OpenBuddy形式のプロンプト生成
    elif PROMPT_TYPE == "openbuddy":
        prefix = f"""You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.{new_line}Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.{new_line}If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.{new_line}You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.{new_line}You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.{new_line}You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.{new_line}{new_line}"""
        messages = curr_system_message + \
            new_line.join([new_line.join([f"User: "+item[0], f"Assistant: "+item[1]])
                    for item in history])
        messages = prefix + messages
    # Airoboros形式のプロンプト生成
    elif PROMPT_TYPE == "airoboros":
        prefix = f"""[INST] <<SYS>>{new_line}You are a helpful, unbiased, uncensored assistant.{new_line}<</SYS>>{new_line}"""
        messages = curr_system_message + \
            f"</s><s>".join(["".join([f"[INST]"+item[0], f"[/INST] "+item[1]])
                    for item in history]).replace(r'[INST]','',1)
        messages = prefix + messages
    # Code Llama形式のプロンプト生成
    elif PROMPT_TYPE == "codellama":
        prefix = f"""[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:{new_line}"""
        messages = curr_system_message + \
            f"</s><s>".join(["".join([f"[INST]"+item[0], f"[/INST]"+item[1]])
                    for item in history]).replace(r'[INST]','',1)
        messages = prefix + messages
    # ELYZA japanese Llama2形式のプロンプト生成
    elif PROMPT_TYPE == "elyzallama2":
        prefix = f"""<s>[INST] <<SYS>>
あなたは誠実で優秀な日本人のアシスタントです。
<</SYS>>{new_line}{new_line}"""
        messages = curr_system_message + \
            f"</s><s>".join(["".join([f"[INST]"+item[0], f"[/INST]"+item[1]])
                    for item in history]).replace(r'[INST]','',1)
        messages = prefix + messages
    # StableBeluga2形式のプロンプト生成
    elif PROMPT_TYPE == "beluga":
        prefix = f"""### System:{new_line}You are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.{new_line}{new_line}"""
        messages = curr_system_message + \
            f"{new_line}{new_line}".join([new_line.join([f"### User:{new_line}"+item[0], f"{new_line}### Assistant:{new_line}"+item[1]])
                    for item in history])
        messages = prefix + messages
    # Japanese StableLM、Nekomata形式のプロンプト生成
    elif PROMPT_TYPE == "ja-stablelm" or PROMPT_TYPE == "nekomata":
        prefix = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。{new_line}{new_line}"""
        messages = curr_system_message + \
            f"{new_line}".join([new_line.join([f"### 指示: "+item[0], f"### 応答: "+item[1]])
                    for item in history])
        messages = prefix + messages
    # Mixtral形式のプロンプト生成
    elif PROMPT_TYPE == "mixtral":
        prefix = f"""<s>"""
        messages = curr_system_message + \
            f"</s><s>".join(["".join([f"[INST]"+item[0], f"[/INST]"+item[1]])
                    for item in history])
        messages = prefix + messages
    # Swallow形式のプロンプト生成
    elif PROMPT_TYPE == "swallow":
        prefix = f"""以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を記述してください。{new_line}{new_line}"""
        messages = curr_system_message + \
            f"{new_line}{new_line}".join([new_line.join([f"### 指示:{new_line}"+item[0], f"{new_line}### 応答:{new_line}"+item[1]])
                    for item in history])
        messages = prefix + messages
    # ELYZA japanese Llama2形式のプロンプト生成
    elif PROMPT_TYPE == "elyzallama2":
        prefix = f"""<s>[INST] <<SYS>>{new_line}あなたは誠実で優秀な日本人のアシスタントです。{new_line}<</SYS>>{new_line}{new_line}"""
        messages = curr_system_message + \
            f"</s><s>".join(["".join([f"[INST]"+item[0], f"[/INST]"+item[1]])
                    for item in history]).replace(r'[INST]','',1)
        messages = prefix + messages
    # KARAKURI LM形式のプロンプト生成
    elif PROMPT_TYPE == "karakuri":
        prefix = f"""<s>[INST] <<SYS>>{new_line}以下の質問やリクエストに対して適切な回答をしてください。{new_line}<</SYS>>{new_line}{new_line}"""
        messages = curr_system_message + \
            f"</s><s>".join(["".join([f"[INST]"+item[0], f"[ATTR] helpfulness: 4 correctness: 4 coherence: 4 complexity: 4 verbosity: 4 quality: 4 toxicity: 0 humor: 0 creativity: 0 [/ATTR] [/INST]"+item[1]])
                    for item in history]).replace(r'[INST]','',1)
        messages = prefix + messages
    # Gemma形式のプロンプト生成
    elif PROMPT_TYPE == "gemma":
        messages = curr_system_message + \
            f"<end_of_turn>model{new_line}".join(["".join([f"<start_of_turn>user{new_line}"+item[0], f"<end_of_turn>{new_line}<start_of_turn>model{new_line}"+item[1]])
                    for item in history])
    # ChatML形式のプロンプト生成
    elif PROMPT_TYPE == "chatml":
        prefix = f"""<|im_start|>system{new_line}以下の質問に日本語で答えてください<|im_end|>{new_line}<|im_start|>"""
        messages = curr_system_message + \
            f"<|im_end|>{new_line}<|im_start|>".join(["".join([f"User{new_line}"+item[0], f"<|im_end|>{new_line}<|im_start|>Assistant{new_line}"+item[1]])
                    for item in history])
        messages = prefix + messages
    # Q&A形式のプロンプト生成
    elif PROMPT_TYPE == "qa":
        messages = curr_system_message + \
            new_line.join([new_line.join(["Q: "+item[0], "A: "+item[1]])
                    for item in history])
    # 特定の書式を使用しない(入力した文章の続きを生成する)場合のプロンプト生成
    elif PROMPT_TYPE == "none":
        messages = curr_system_message + \
            "".join(["".join([item[0], item[1]])
                    for item in history])
    # PROMPT_TYPE設定が正しくなければ終了する
    else:
        print(f"Invalid PROMPT_TYPE \"{PROMPT_TYPE}\"")
        exit()
    # 生成したプロンプト文字列を返す
    return messages


def chat(curr_system_message, history, p_temperature, p_top_k, p_top_p, p_repetition_penalty, p_max_new_tokens):

    # 会話履歴を表示
    if DEBUG_FLAG:
        print(f"history={history}\n")

    # プロンプト文字列生成
    del_flag = 0
    while True:
        # プロンプト文字列を生成する
        messages = prompt(curr_system_message, history)
        # もしプロンプトの文字数が多すぎる場合は削除フラグを設定
        if del_flag == 0 and len(messages) > PROMPT_THRESHOLD:
            del_flag = 1
        # 削除フラグが設定され、かつPROMPT_DELETEDより文字数が多い場合は履歴の先頭を削除
        if del_flag == 1 and len(messages) > PROMPT_DELETED:
            history.pop(0)
        # 削除フラグが設定されてないか、設定されているがPROMPT_DELETEDよりトークン数が少ない場合ループを抜ける
        else:
            break

    # プロンプトを標準出力に表示
    if DEBUG_FLAG:
        print(f"--prompt strings--\n{messages}\n----\n")

    # モデルに入力して回答を生成(ストリーミング出力させる)
    streamer = m.create_completion(
                   messages,
                   max_tokens=p_max_new_tokens,
                   temperature=p_temperature,
                   top_k=p_top_k,
                   top_p=p_top_p,
                   repeat_penalty=p_repetition_penalty,
                   stream=True,
                   stop=["</s>","<|im_end|>"],
               )

    #print(history)
    # Initialize an empty string to store the generated text
    partial_text = ""
    for msg in streamer:
        temp = msg['choices'][0]
        if 'text' in temp:
            #print(temp['text'])
            partial_text += temp['text']
            history[-1][1] = partial_text
            # Yield an empty string to cleanup the message textbox and the updated conversation history
            yield history
    if DEBUG_FLAG:
        print(f"--generated strings--\n{partial_text}\n----\n")
    #return partial_text
    return history


#------
# 実行
#------

# 引数を取得
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=BASE_MODEL, help="モデル名またはディレクトリのパス")
parser.add_argument("--model-type", type=str, choices=["llama"],  default=MODEL_TYPE, help="モデルタイプ名")
parser.add_argument("--lora", type=str, default=LORA_WEIGHTS, help="LoRAのパス")
parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE, help="コンテキストサイズ")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="バッチサイズ")
parser.add_argument("--gpu", type=str, choices=["on", "off"], default=USE_GPU, help="GPUを使用するかどうか(使用にはcublas対応のllama-cpp-pythonが必要)")
parser.add_argument("--gpu-layers", type=int, default=GPU_LAYERS, help="GPUで処理するレイヤ数")
parser.add_argument("--tensor-split", type=str, default=TENSOR_SPLIT, help="複数GPUでの分割指定(メモリ使用量の比をカンマ区切りの少数で指定する)")
parser.add_argument("--threads", type=int, default=THREAD_NUM, help="使用するCPUコア数")
parser.add_argument("--use-mmap", type=str, choices=["on", "off"], default=USE_MMAP, help="mmapが使用可能な場合に使用するか")
parser.add_argument("--use-mlock", type=str, choices=["on", "off"], default=USE_MLOCK, help="mlockを使用するか")
parser.add_argument("--prompt-type", type=str, choices=["rinna", "vicuna", "alpaca", "llama2", "openbuddy", "airoboros", "codellama", "elyzallama2", "beluga", "ja-stablelm", "mixtral", "swallow", "nekomata", "elyzallama2", "karakuri", "gemma", "chatml", "qa", "none"], default=PROMPT_TYPE, help="プロンプトタイプ名")
parser.add_argument("--prompt-threshold", type=int, default=PROMPT_THRESHOLD, help="このトークン数を超えたら古い履歴を削除")
parser.add_argument("--prompt-deleted", type=int, default=PROMPT_DELETED, help="古い履歴削除時にこのトークン以下にする")
parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY, help="繰り返しに対するペナルティ")
parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="推論時に生成するトークン数の最大")
parser.add_argument("--setting-visible", type=str, choices=["on", "off"], default=SETTING_VISIBLE, help="詳細設定を表示するかどうか")
parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="生成する文章の多様さ")
parser.add_argument("--rope-freq-base", type=float, default=ROPE_BASE, help="Rope SamplingのBase Frequency")
parser.add_argument("--rope-freq-scale", type=float, default=ROPE_SCALE, help="Rope ScalingのSacle Factor")
parser.add_argument("--host", type=str, default=GRADIO_HOST, help="WebサーバがバインドするIPアドレスorホスト名")
parser.add_argument("--port", type=int, default=GRADIO_PORT, help="Webサーバがバインドするポート番号")
parser.add_argument("--title", type=str, default=TITLE_STRINGS, help="Webページのタイトル")
parser.add_argument("--debug", type=str, choices=["on", "off"], default=DEBUG_FLAG, help="デバッグメッセージを標準出力に表示")
args = parser.parse_args()

# 引数でセットされた値で上書きする
BASE_MODEL = args.model
MODEL_TYPE = args.model_type
LORA_WEIGHTS = args.lora
CONTEXT_SIZE = args.context_size
BATCH_SIZE = args.batch_size
USE_GPU = args.gpu
GPU_LAYERS = args.gpu_layers
THREAD_NUM = args.threads
USE_MMAP = args.use_mmap
USE_MLOCK = args.use_mlock
PROMPT_TYPE = args.prompt_type
PROMPT_THRESHOLD = args.prompt_threshold
PROMPT_DELETED = args.prompt_deleted
REPETITION_PENALTY=args.repetition_penalty
MAX_NEW_TOKENS = args.max_new_tokens
SETTING_VISIBLE = args.setting_visible
TEMPERATURE = args.temperature
ROPE_BASE = args.rope_freq_base
ROPE_SCALE = args.rope_freq_scale
GRADIO_HOST = args.host
GRADIO_PORT = args.port
TITLE_STRINGS = args.title
DEBUG_FLAG = args.debug

# tensor_splitの処理
if args.tensor_split == None or args.tensor_split == "None" or args.tensor_split == "none":
    TENSOR_SPLIT = None
else:
    if re.match(r'^[0-9,\.]+$',args.tensor_split):
        temp_list = args.tensor_split.split(',')
        TENSOR_SPLIT = [float(s) for s in temp_list]
    else:
        print("Invalid TENSOR_SPLIT value.")
        exit()

# パラメータ表示
print("---- パラメータ ----")
print(f"モデル名orパス: {BASE_MODEL}")
print(f"モデルタイプ名: {MODEL_TYPE}")
if LORA_WEIGHTS == "":
    print(f"LoRAモデルパス: (LoRAなし)")
else:
    print(f"LoRAモデルパス: {LORA_WEIGHTS}")
print(f"コンテキストサイズ: {CONTEXT_SIZE}")
print(f"バッチサイズ: {BATCH_SIZE}")
print(f"GPU使用: {USE_GPU}")
print(f"GPUレイヤ数: {GPU_LAYERS}")
print(f"GPU分割指定: {TENSOR_SPLIT}")
print(f"使用CPUコア数: {THREAD_NUM}")
print(f"mmap使用: {USE_MMAP}")
print(f"mlock使用: {USE_MLOCK}")
print(f"プロンプトタイプ: {PROMPT_TYPE}")
print(f"プロンプトトークン数しきい値: {PROMPT_THRESHOLD}")
print(f"プロンプトトークン数削除値: {PROMPT_DELETED}")
print(f"繰り返しペナルティ: {REPETITION_PENALTY}")
print(f"生成最大トークン数: {MAX_NEW_TOKENS}")
print(f"詳細設定表示: {SETTING_VISIBLE}")
print(f"Temperature: {TEMPERATURE}")
print(f"Rope Sampling Frequency: {ROPE_BASE}")
print(f"Rope Sampling Scale: {ROPE_SCALE}")
print(f"WebサーバIPorホスト名: {GRADIO_HOST}")
print(f"Webサーバポート番号: {GRADIO_PORT}")
print(f"Webページタイトル: {TITLE_STRINGS}")
print(f"デバッグ: {DEBUG_FLAG}\n")

# LORA_WEIGHTSが指定されていなければNoneをセット
if LORA_WEIGHTS == "":
    LORA_WEIGHTS == None
# USE_GPUはTrue or Falseに変換
if USE_GPU == "on":
    USE_GPU = True
else:
    USE_GPU = False
# USE_MMAPはTrue or Falseに変換
if USE_MMAP == "on":
    USE_MMAP = True
else:
    USE_MMAP = False
# USE_MLOCKはTrue or Falseに変換
if USE_MLOCK == "on":
    USE_MLOCK = True
else:
    USE_MLOCK = False
# SETTING_VISIBLEはTrue or Falseに変換
if SETTING_VISIBLE == "on":
    SETTING_VISIBLE = True
else:
    SETTING_VISIBLE = False
# DEBUG_FLAGはTrue or Falseに変換
if DEBUG_FLAG == "on":
    DEBUG_FLAG = True
else:
    DEBUG_FLAG = False
# スレッド数が0の場合Noneに置き換える
if THREAD_NUM == 0:
    THREAD_NUM = None


## モデルタイプによる設定とモデルのロード

# Llama系モデルの場合
if MODEL_TYPE == "llama":
    from llama_cpp import Llama
    # 改行を示す文字の設定
    new_line = "\n"
    # モデルのロード
    print(f"Starting to load the model \"{BASE_MODEL}\" to memory")
    # GPUで処理する場合
    if USE_GPU:
        m = Llama(
              model_path=BASE_MODEL,
              lora_path=LORA_WEIGHTS,
              n_ctx=CONTEXT_SIZE,
              n_threads=THREAD_NUM,
              n_gpu_layers=GPU_LAYERS,
              rope_freq_base=ROPE_BASE,
              rope_freq_scale=ROPE_SCALE,
              tensor_split=TENSOR_SPLIT,
              n_batch=BATCH_SIZE,
            )
    # CPUで処理する場合
    else:
        m = Llama(
              model_path=BASE_MODEL,
              lora_path=LORA_WEIGHTS,
              n_ctx=CONTEXT_SIZE,
              n_threads=THREAD_NUM,
              rope_freq_base=ROPE_BASE,
              rope_freq_scale=ROPE_SCALE,
              use_mmap=USE_MMAP,
              use_mlock=USE_MLOCK,
            )
    print(f"Sucessfully loaded the model to the memory")
# MODEL_TYPE設定が正しくなければ終了する
else:
    print(f"Invalid MODEL_TYPE \"{MODEL_TYPE}\"")
    exit()

# プロンプトの先頭に付加する文字列
start_message = ""

# Webページ
with gr.Blocks(title="GGUF Simple WebUI", theme=gr.themes.Base()) as demo:
    history = gr.State([])
    gr.Markdown(f"## {TITLE_STRINGS}")
    chatbot = gr.Chatbot(height=500)
    with gr.Row():
        with gr.Column(scale=20):
            msg = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box",
                             show_label=False, container=False)
        with gr.Column(scale=1, min_width=100):
            submit = gr.Button("Submit")
    with gr.Row():
                stop = gr.Button("Stop")
                regenerate = gr.Button("Regenerate")
                removelast = gr.Button("Remove last")
                clear = gr.Button("Clear")
    with gr.Accordion(label="Advanced Settings", open=False, visible=SETTING_VISIBLE):
        with gr.Blocks():
            p_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=TEMPERATURE, step=0.1, label="Temperature", interactive=True)
            p_top_k = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Top_K (0=無効)", interactive=True)
            p_top_p = gr.Slider(minimum=0.01, maximum=1.00, value=1.00, step=0.01, label="Top_P (1.00=無効)", interactive=True)
        with gr.Blocks():
            p_max_new_tokens = gr.Slider(minimum=1, maximum=4096, value=MAX_NEW_TOKENS, step=1, label="Max New Tokens", interactive=True)
            p_repetition_penalty = gr.Slider(minimum=1.00, maximum=5.00, value=REPETITION_PENALTY, step=0.01, label="Repetition Penalty (1.00=ペナルティなし)", interactive=True)

    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False)

    submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot, p_temperature, p_top_k, p_top_p, p_repetition_penalty, p_max_new_tokens], outputs=[chatbot], queue=True)

    submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot, p_temperature, p_top_k, p_top_p, p_repetition_penalty, p_max_new_tokens], outputs=[chatbot], queue=True)

    regenerate_click_event = regenerate.click(fn=regen, inputs=[chatbot], outputs=[msg, chatbot], queue=False).then(
               lambda: None, None, [msg], queue=False).then(
                   fn=chat, inputs=[system_msg, chatbot, p_temperature, p_top_k, p_top_p, p_repetition_penalty, p_max_new_tokens], outputs=[chatbot], queue=True)

    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, submit_click_event, regenerate_click_event], queue=False)

    removelast.click(fn=remove_last, inputs=[chatbot], outputs=[chatbot], queue=False)

    clear.click(lambda: None, None, [chatbot], queue=False)

demo.queue(max_size=32, concurrency_count=2)
demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, share=False)
