import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设置模型路径（可以是您本地下载好的文件夹绝对路径，也可以是hf的repo名）
model_id = "zemelee/qwen2.5-jailbreak"

print("正在加载模型和分词器...")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 加载模型 (根据您之前的日志，您使用了 bfloat16，这里保持一致可以省显存)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 自动分配到可用的显卡上
    trust_remote_code=True
)
print("模型加载完成！输入 'quit' 退出。")

# 2. 开启多轮对话循环
messages =[]
while True:
    user_input = input("\n用户: ")
    if user_input.lower() == 'quit':
        break

    messages.append({"role": "user", "content": user_input})

    # 使用模型自带的聊天模板进行拼接
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 转为 tensor 并放到显卡上
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回复
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,      # 最大生成长度
        temperature=0.7,         # 创造性，越高越随机
        top_p=0.9
    )

    # 截取新生成的部分（去掉输入提示词）
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码成文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"\n模型: {response}")

    # 将模型的回复也加入历史记录，以便进行多轮对话
    messages.append({"role": "assistant", "content": response})