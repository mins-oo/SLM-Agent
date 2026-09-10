Fine tuning **qwen2.5-coder:3b** model with Korean dataset **nlpai-lab/kullm-v2**
https://huggingface.co/datasets/nlpai-lab/kullm-v2

```
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# ------------------------------------------
# 1. 모델 및 토크나이저 로드 (Unsloth Qwen2.5-Coder-3B 4bit)
# ------------------------------------------
max_seq_len = 2048
dtype = None  # GPU/iGPU 사양에 따라 bfloat16/float16 자동 선택
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
    max_seq_length = max_seq_len,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

# ------------------------------------------
# 2. LoRA 어댑터 설정
# ------------------------------------------
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

# ------------------------------------------
# 3. 데이터셋 로드 및 Qwen 대화 템플릿(ChatML) 변환
# ------------------------------------------
dataset = load_dataset("nlpai-lab/kullm-v2", split="train")

def format_prompts(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []

    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # input 필드가 존재하는 경우 지시문에 통합
        if input_text and len(input_text.strip()) > 0:
            user_content = f"{instruction}\n\n[입력]:\n{input_text}"
        else:
            user_content = instruction

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]

        # Qwen 전용 ChatML 특수 태그(<|im_start|> 등) 형태로 변환
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False
        )
        texts.append(text)

    return {"text": texts}

dataset = dataset.map(format_prompts, batched = True)

# ------------------------------------------
# 4. SFTTrainer 설정 및 학습 시작
# ------------------------------------------
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_len,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 테스트용 step 수 (실제 전수 학습 시 num_train_epochs = 1 권장)
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

# ------------------------------------------
# 5. GGUF 포맷으로 저장
# ------------------------------------------
# 4-bit q4_k_m 양자화 저장 (내장 그래픽/iGPU 환경에 가장 권장)
model.save_pretrained_gguf(
    "qwen2.5_coder_kullm_q4_k_m",
    tokenizer,
    quantization_method = "q4_k_m"
)
```