# GGUF Simple WebUI
llama-cpp-python(llama.cpp)で実行するGGUF形式のLLM用の簡易Webチャットインタフェースです。

# 機能
- ブラウザで開いたWebインタフェース上で、LLMとのチャットができます
- ストリーミング(生成中の表示)に対応
- GPUを搭載していない環境でも実行可能 (処理速度はGPU使用時より遅いです)

# 動作要件
- Rocky Linux 8.8 上の Python 3.9.13 の環境で本スクリプトを作成していますが、おそらく [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)、[gradio](https://github.com/gradio-app/gradio) モジュールがインストールされたPython環境があれば動作すると思います
- GPUはNVIDIA GPU、CUDA 環境で確認しています (GeForce RTX 3060、CUDA 11.7 環境で作成しています)
- GPUを使用するには、llama-cpp-pythonがGPU対応するようインストールする必要があります。インストール方法は [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) を参照してください。

# 使い方

## 起動方法

- `gguf-webui.py` 内の15行目以降に各種設定項目がありますので、実行したいモデル、使用したいプロンプト、WebUIで使用するIPアドレスや ポート番号などを記述して起動してください
```bash
$ python3 gguf-webui.py
```
`gguf-webui.py` のファイル名に指定はないため、ファイルを任意の名前でコピーして、モデルごとや設定ごとに使い分けることができます

- 実行時のオプションで `gguf-webui.py` 内の設定を上書きできるため、コマンドオプションのみで設定を指定して起動することも可能です

- 実行前にGGUF形式のモデルを入手して任意のディレクトリ内に配置してください

Vicuna 13B v1.5モデルでの実行コマンド例
```bash
$ python3 gguf-webui.py \
        --model ~/GGUF_Model/vicuna-13b-v1.5.Q4_K_M.gguf \
        --gpu on \
        --gpu-layers 30 \
        --max-new-tokens 1024 \
        --prompt-type vicuna \
        --prompt-threshold 3072 \
        --prompt-deleted 3071 \
        --title "Vicuna 13B v1.5 GGUF Chat"
```

Vicuna 13B v1.5 16kモデルでの実行コマンド例 (コンテキスト拡張されているため、Rope Scalingの設定が必要)
```bash
$ python3 llm-webui.py \
        --model ~/GGUF_Model/vicuna-13b-v1.5-16k.Q4_K_M.gguf \
        --gpu on \
        --gpu-layers 30 \
        --context-size 16384 \
        --max-new-tokens 4096 \
        --rope-freq-base 10000.0 \
        --rope-freq-scale 0.25 \
        --prompt-type vicuna \
        --prompt-threshold 12288 \
        --prompt-deleted 12287 \
        --title "Vicuna 13B v1.5 16K GGUF Chat"
```

- 起動したら、ブラウザで http://127.0.0.1:7860 (IPアドレス、ポート番号を変更した場合はそれに合わせてください)を開いてください

## 設定

以下のオプションが指定可能です。

| オプション                              | 説明                                                            |
| ------------------------------------ | ----------------------------------------------------------------- |
| --help                               | 指定可能なオプションの情報を出力する                                |
| --model <モデルファイルのパス>             | 保存したモデルファイルのパスを指定する   |
| --model-type <モデルタイプ名>        | モデルタイプを指定する。現時点では `llama` のみのためあえて指定する必要はない |
| --context-type <プロンプトタイプ名>   | モデルが扱えるコンテキストサイズを指定する。 |
| --gpu <on/off>   | GPUを使用するかどうかを指定する。 |
| --gpu-layers <レイヤ数>   | GPUを使用する場合、GPUで処理を行うレイヤ数を指定する。この値が大きいほどGPUメモリが必要になる。 |
| --tensor-split <使用させたいGPUメモリ量の比率>   | 複数GPUが利用可能な場合、カンマ区切りでそれぞれのGPUが使用するメモリ量の比率を `0.3,0,7` のように指定する。 |
| --threads <スレッド数>   | CPUコアをいくつ使用して処理するかを指定する。 |
| --use-mmap <on/off>   | CPUで処理する場合、MMAPを使用するかどうかを指定する。 |
| --use-mlock <on/off>   | CPUで処理する場合、メインメモリ上にモデルを常駐させるかどうかを指定する。 |
| --prompt-type <プロンプトタイプ名>   | 使用するプロンプトテンプレートを指定する。基本的にはモデルの学習に使用されたテンプレートを指定する。詳細は下の項目を参照 |
| --prompt-threshold <トークン数>      | プロンプト生成時に会話履歴を含めたトークン数がここで設定した数を超えると会話履歴が古い順に削除される |
| --prompt-deleted <トークン数>        | `--prompt-threshold` 設定値を超えて会話履歴が削除される場合、ここで指定したトークン数以 下になるまで削除される |
| --max-new-tokens <トークン数>        | モデルが一度に生成する最大トークン数 |
| --temperature <Temperature値>        | 値を大きくすると多様な出力を行うようになる。0～1の間の値を設定する |
| --repetition-penalty <繰り返しペナルティ値>        | 値を大きくすると、繰り返しが発生しにくくなる。1.0だとペナルティなしとなる |
| --setting-visible <on/off>        | Advanced SettingsをWebUI上に表示するかどうか |
| --rope-freq-base <Rope Scaling Base値>      | `10000.0` を指定しておけば大丈夫そう |
| --rope-freq-scale <Rope Scaling 値>      | llama2ベースのモデルの場合、コンテキストサイズが4kのモデルの場合は `1.0` を、コンテキストサイズが16kのモデルの場合は `0.25` を指定しておけば大丈夫そう |
| --host <IPアドレス>                  | WebUIがバインドするアドレスを指定する。同じPC上のブラウザから使用する場合は `127.0.0.1` でよい |
| --port <ポート番号>                  | WebUIがバインドするポート番号を指定する。他のプログラムが使用していなければいくつでもよ いが、Linux上で実行する場合、1024以下を指定するには通常root権限が必要 |
| --title "<タイトル文字列>"           | WebUIの最上部に表示するタイトルを任意に指定可能 |
| --debug <on/off>                     | コンソールにデバッグ情報を表示 |


## プロンプトタイプについて

主に、日本語対応のモデルで使用されるプロンプトが使用可能です。
モデルの学習時に使用されたプロンプトを指定することで精度の高い回答が期待できます。

### プロンプトタイプ名 `none`
プロンプトは使用せず、ユーザの入力した続きの文章をモデルが出力する形式。ファインチューニング前提の素のモデルを試す時などに使う
```
<ユーザの入力した文章><モデルの出力した文章>
```


### プロンプトタイプ名 `vicuna`
プロンプト形式
```
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: <ユーザの入力した文章>
ASSISTANT: <モデルの出力した文章>
```


### プロンプトタイプ名 `alpaca`
プロンプト形式
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
<ユーザの入力した文章>

### Response:
<モデルの出力した文章>
```


### プロンプトタイプ名 `llama2`
プロンプト形式
```
System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
User: <ユーザの入力した文章>
Assistant: <モデルの出力した文章>
```


### プロンプトタイプ名 `openbuddy`
プロンプト形式
```
You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.
Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.
You cannot access theinternet, but you have vast knowledge, cutoff: 2021-09.
You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.

User: <ユーザの入力した文章>
Assistant: <モデルの出力した文章>
```


### プロンプトタイプ名 `airoboros`
プロンプト形式
```
[INST] <<SYS>>
You are a helpful, unbiased, uncensored assistant.
<</SYS>>
<ユーザの入力した文章>[/INST]<モデルの出力した文章>
```


### プロンプトタイプ名 `codellama`
プロンプト形式
```
[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
<ユーザの入力した文章>
[/INST]<モデルの出力した文章>
```


### プロンプトタイプ名 `elyzallama2`
プロンプト形式
```
<s>[INST] <<SYS>>
あなたは誠実で優秀な日本人のアシスタントです。
<</SYS>>

<ユーザの入力した文章>[/INST]<モデルの出力した文章></s>
```


### プロンプトタイプ名 `beluga`
プロンプト形式
```
### System:
You are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.

### User:
<ユーザの入力した文章>

### Assistant:
<モデルの出力した文章>
```


### プロンプトタイプ名 `ja-stablelm`
プロンプト形式
```
以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい

### 指示: <ユーザの入力した文章>
### 応答: <モデルの出力した文章>
```


### プロンプトタイプ名 `qa`
プロンプト形式
```
Q: <ユーザの入力した文章>
A: <モデルの出力した文章>
```

# ライセンス

Japanese LLM Simple WebUI is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).