import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from utils import create_gguf

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

instruction = 'Read the job vacancy text and extract the following fields in JSON format, adhering strictly to the provided structure:\n\n{\n   "title": string,               // Job title.\n   "overview": string,            // Job Summary.\n   "expectations": [string],      // Array of expectations from the candidate (e.g., "A minimum of 5+ years of experience", "Strong communication skills", "Team player").\n   "employment_types": [string],  // Use predefined values.\n   "experience_levels": [string], // Use predefined values.\n   "must_have_skills": [string],  // Essential skills (e.g., "React", "Python", "SQL").\n   "optional_skills": [string],   // Non-critical skills (e.g., "Node.js", "Vue.js").\n   "relocation_assistance": bool, // True if relocation assistance is mentioned; otherwise, False.\n   "visa_sponsorship": bool,      // True if visa sponsorship is offered; otherwise, False.\n   "locations": [                 // Array of location objects, even if there is only one location.\n      {\n         "presence": string,       // Use predefined values.\n         "country_code": string,   // ISO country code (e.g., "US").\n         "region_code": string,    // Predefined region code.\n         "subregion_code": string, // Predefined subregion code.\n         "city": string,           // Name of the city (e.g., "New York").\n         "timezone": string,       // Timezone in IANA format (e.g., "America/New_York").\n         "visa_package": bool,     // True if a visa package is provided; otherwise, False.\n         "relocation_package": bool // True if a relocation package is provided; otherwise, False.\n      }\n   ],\n   "foreign_languages": [string], // Required foreign languages (e.g., "English", "German").\n   "is_stock_options_available": bool, // True if stock options are mentioned; otherwise, False.\n   "salary": {                     // Salary object or null if not mentioned.\n      "min": number,               // Minimum salary amount.\n      "max": number,               // Maximum salary amount.\n      "currency": string,          // Currency code (e.g., "USD", "EUR").\n      "interval": string           // Predefined payment interval.\n   },\n   "categories": [string]          // Job categories or domains (e.g., "AI_MACHINE_LEARNING", "BLOCKCHAIN").\n}\n\nFor missing fields, use the following default values:\n- Boolean fields: `false`\n- String fields: an empty string or `null` (if explicitly stated in the JSON structure).\n- Arrays: an empty array `[]`.\n- Objects: `null` if all subfields are missing.\n'
input = "Regional Sales Director.\n\nLocation: Remote, Andorra la Vella, AD.\n\nCompensation: Competitive salary package, including stock options, based on experience.\n\nAbout the Company:\nAt [Company Name], we are dedicated to revolutionizing the [industry/field] with innovative solutions that drive growth and success. Our team of experts is passionate about delivering exceptional results and building strong relationships with our customers and partners. We are seeking a highly motivated and experienced Regional Sales Director to lead our sales team in the region and drive revenue growth.\n\nRole Overview:\nAs a Regional Sales Director, you will be responsible for developing and executing sales strategies to achieve business objectives, leading and managing sales teams, and building and maintaining strong relationships with key customers and partners. You will work closely with cross-functional teams to drive sales initiatives and analyze sales data to identify trends and areas for improvement.\n\nResponsibilities:\n\n* Develop and execute sales strategies to achieve business objectives, including setting sales targets, identifying new business opportunities, and negotiating contracts.\n* Lead and manage sales teams to drive revenue growth, including coaching, mentoring, and developing team members to achieve their full potential.\n* Build and maintain strong relationships with key customers and partners, including identifying and addressing their needs, and providing exceptional customer service.\n* Collaborate with cross-functional teams, including marketing, product, and operations, to drive sales initiatives and ensure alignment with business objectives.\n* Analyze sales data to identify trends and areas for improvement, and develop and implement strategies to address these areas.\n\nRequirements:\n\n* 5+ years of experience in sales leadership, with a track record of achieving and exceeding sales targets.\n* Strong understanding of sales strategies, tactics, and methodologies, with the ability to develop and execute effective sales plans.\n* Excellent communication, interpersonal, and leadership skills, with the ability to build and maintain strong relationships with customers, partners, and team members.\n* Proficiency in Microsoft Office, including Excel, Word, and PowerPoint.\n* Bachelor's degree in Business Administration, Marketing, or a related field.\n\nOptional Skills:\n\n* Sales analytics, including data analysis and market research.\n* Customer relationship management, including CRM software and sales automation tools.\n\nAdditional Information:\nAs a remote worker, you will have the flexibility to work from anywhere, as long as you have a reliable internet connection and a quiet, dedicated workspace. You will also have access to a range of benefits, including stock options, health insurance, and a comprehensive training program.\n\nImportant Notes:\n\n* This role is not eligible for relocation assistance or visa sponsorship.\n* The successful candidate will be required to provide proof of eligibility to work in Andorra.\n* The company offers a range of benefits, including health insurance, stock options, and a comprehensive training program."

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

huggingface_model_name = "vchesheiko/Meta-Llama-3.1-8B-Instruct-bnb-4bit-Jobber-1.0.0"

# 2. Before training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=HF_TOKEN,
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction,  # instruction
            input,  # input
            "",  # output - leave this blank for generation!
        )
    ],
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)


# 3. Load data

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


pass
url = "https://huggingface.co/datasets/vchesheiko/job_desc_json_data/resolve/main/alpaca_jobs_dataset.json"
dataset = load_dataset("json", data_files={"train": url}, split="train", token=HF_TOKEN)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

# 4. Training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,  # Set this for 1 full training run.
        # max_steps=100,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
    ),
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# 5. After Training
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction,  # instruction
            input,  # input
            "",  # output - leave this blank for generation!
        )
    ],
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)

# 6. Saving
model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
model.push_to_hub(huggingface_model_name, token=HF_TOKEN)
tokenizer.push_to_hub(huggingface_model_name, token=HF_TOKEN)

# Merge to 16bit
if True:
    model.save_pretrained_merged(
        "model",
        tokenizer,
        save_method="merged_16bit",
    )
if True:
    model.push_to_hub_merged(
        huggingface_model_name,
        tokenizer,
        save_method="merged_16bit",
        token=HF_TOKEN,
    )

# # Merge to 4bit
# if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "merged_4bit", token = HF_TOKEN)

# # Just LoRA adapters
# if True: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
# if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "lora", token = HF_TOKEN)

# # Save to 8bit Q8_0
# if True: model.save_pretrained_gguf("model", tokenizer,)
# if True: model.push_to_hub_gguf(huggingface_model_name, tokenizer, token = HF_TOKEN)

# # Save to 16bit GGUF
# if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if True: model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "f16", token = HF_TOKEN)

# # Save to q4_k_m GGUF
# if True:
#     model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
# if True:
#     model.push_to_hub_gguf(
#         huggingface_model_name, tokenizer, quantization_method="q4_k_m", token=HF_TOKEN
#     )

# # Save to multiple GGUF options - much faster if you want multiple!
# if True:
#     model.push_to_hub_gguf(
#         huggingface_model_name, # Change hf to your username!
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
#         token = HF_TOKEN
#     )

# Save and push gguf
create_gguf("lora_model", "q4_k_m", huggingface_model_name)
