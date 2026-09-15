**qwen2.5-coder:3b** 모델의 한국어 능력을 향상시키려 **nlpai-lab/kullm-v2** 데이터셋으로 파인튜닝을 시도해보았다.
coder 모델 특성상 대화 용도로 파인튜닝을 하니 성능이 저하되는 것을 느껴 **llama3.2:3b** 모델로 변경

<details>
<summary>fine tuning + backup on drive</summary>

```
import os
import torch
from google.colab import drive
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 1. 구글 드라이브 마운트 & 경로 설정
# ==========================================
drive.mount('/content/drive')

# 구글 드라이브 내 저장 폴더 설정
drive_base_dir = "/content/drive/MyDrive/FineTune"
checkpoint_dir = os.path.join(drive_base_dir, "checkpoints")
gguf_output_dir = os.path.join(drive_base_dir, "gguf_output")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(gguf_output_dir, exist_ok=True)

# ==========================================
# 2. 모델 & 토크나이저 로드
# ==========================================
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# Llama 3 전용 Chat Template 적용
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# ==========================================
# 3. LoRA 가중치 설정
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 4. 데이터셋 로드 & 포맷팅
# ==========================================
print("kullm-v2 데이터셋 로드 중...")
dataset = load_dataset("nlpai-lab/kullm-v2", split="train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        user_content = f"{instruction}\n\n{input_text}".strip() if input_text else instruction.strip()
            
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.strip()}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 5. Trainer & 체크포인트 저장 설정
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.05,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        
        # --- 드라이브 자동 저장 설정 ---
        output_dir = checkpoint_dir,
        save_strategy = "steps",
        save_steps = 500,            # 500 스텝마다 체크포인트 구글 드라이브 저장
        save_total_limit = 2,        # 용량 관리를 위해 최근 2개 체크포인트만 유지
    ),
)

# ==========================================
# 6. 이전 체크포인트 감지 및 학습 실행
# ==========================================
last_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoints = [
        os.path.join(checkpoint_dir, d) 
        for d in os.listdir(checkpoint_dir) 
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        # 스텝 번호가 가장 높은 마지막 체크포인트 탐색
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

if last_checkpoint:
    print(f"이전 체크포인트를 감지했습니다: {last_checkpoint}")
    print("연결이 끊기기 전 시점부터 학습을 재개합니다...")
    trainer.train(resume_from_checkpoint = last_checkpoint)
else:
    print("저장된 체크포인트가 없습니다. 처음부터 학습을 시작합니다...")
    trainer.train()

# ==========================================
# 7. Q8_0 GGUF 변환 및 구글 드라이브 저장
# ==========================================
print(f"GGUF (Q8_0) 변환 및 드라이브 저장 중... (경로: {gguf_output_dir})")
model.save_pretrained_gguf(
    gguf_output_dir,
    tokenizer,
    quantization_method = "q8_0"
)

print("파인튜닝 및 Q8_0 GGUF 파일 추출 완료")
```
</details>
