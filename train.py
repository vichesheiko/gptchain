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

instruction = "Read the job vacancy text and extract the following fields in JSON format: title, overview, qualifications, employment_types, experience_levels, must_have_skills, optional_skills, relocation_assistance, visa_sponsorship, locations, foreign_languages, is_stock_options_available, salary, categories. For the 'overview' field, create a value with basic job description that includes only 700-1000 words. For the 'qualifications' field, create a value that includes only 700-1000 words. For the 'must_have_skills' and 'optional_skills' fields, create arrays with skills. Each skill should consist of no more than three words, and only the most essential skill variations should be included (e.g., 'React', not 'React JS Developer' or 'React.JS expert'). For the field 'employment_types' use only the predefined values: CONTRACT, FULL_TIME, PART_TIME, INTERNSHIP, VOLUNTEER, OTHER (others are not allowed). For the field 'experience_levels' use only the predefined values: INTERNSHIP, JUNIOR, MIDDLE, SENIOR, LEAD, ARCHITECT, MANAGER, DIRECTOR, EXECUTIVE, STAFF (others are not allowed). For the field 'categories' use only the predefined values: ACCOUNT_MANAGEMENT, AI_MACHINE_LEARNING, AR_VR, BACKEND_DEVELOPMENT, BLOCKCHAIN, BUSINESS_INTELLIGENCE, CYBERSECURITY, DATABASE_ADMINISTRATION, DATA_SCIENCE, DESIGN_ENGINEER, DEVELOPER_RELATIONS, DEVOPS_CLOUD, DIGITAL_MARKETING, FINANCE, FRONTEND_DEVELOPMENT, FULL_STACK_DEVELOPMENT, GAME_DESIGN, GAME_DEVELOPMENT, HARDWARE_ENGINEERING, IOT, IT_SYSTEM_ADMINISTRATION, LEGAL, MOBILE_DESIGN, MOBILE_DEVELOPMENT, NETWORK_ENGINEERING, OTHER, OTHER_DESIGN, PRODUCT_MANAGEMENT, PRODUCT_MARKETING, PROJECT_MANAGEMENT, PUBLIC_RELATIONS, RESEARCH_DEVELOPMENT, ROBOTICS, TECHNICAL_SUPPORT, TELECOMMUNICATIONS_ENGINEERING, TESTING_QA, WEB_DESIGN, OTHER. For the field 'salary', include the optionally subfields: min (minimum salary), max (maximum salary), currency (currency code) and interval (payment interval). If the salary details are not provided, leave the field empty. For the field 'locations', include the optionally subfields: presence (predefined value) city (city name) country_code (predefined value), subregion_code (predefined value), region_code (predefined value), timezone (IANA format), visa_package (If there is no visa package, specify false) and relocation_package (If there is no relocation package, specify false). For the field 'salary.interval' use only the predefined values: FIXED, ANNUAL, MONTHLY, WEEKLY, DAILY, HOURLY (others are not allowed). For the field 'locations.presence' use only the predefined values: OFFICE, HYBRID, REMOTE (others are not allowed). For the field 'locations.country_code' use only the predefined values: AD (Andorra), AE (United Arab Emirates), AF (Afghanistan), AG (Antigua and Barbuda), AI (Anguilla), AL (Albania), AM (Armenia), AO (Angola), AR (Argentina), AS (American Samoa), AT (Austria), AU (Australia), AW (Aruba), AZ (Azerbaijan), BA (Bosnia and Herzegovina), BB (Barbados), BD (Bangladesh), BE (Belgium), BF (Burkina Faso), BG (Bulgaria), BH (Bahrain), BI (Burundi), BJ (Benin), BL (Saint Barthelemy), BM (Bermuda), BN (Brunei Darussalam), BO (Bolivia), BR (Brazil), BS (Bahamas), BT (Bhutan), BW (Botswana), BY (Belarus), BZ (Belize), CA (Canada), CC (Cocos (Keeling) Islands), CD (Congo (Democratic Republic)), CF (Central African Republic), CG (Congo), CH (Switzerland), CK (Cook Islands), CL (Chile), CM (Cameroon), CN (China), CO (Colombia), CR (Costa Rica), CU (Cuba), CV (Cape Verde), CX (Christmas Island), CY (Cyprus), CZ (Czech Republic), DE (Germany), DJ (Djibouti), DK (Denmark), DM (Dominica), DO (Dominican Republic), DZ (Algeria), EC (Ecuador), EE (Estonia), EG (Egypt), EH (Western Sahara), ER (Eritrea), ES (Spain), ET (Ethiopia), FI (Finland), FJ (Fiji), FK (Falkland Islands), FM (Micronesia), FO (Faroe Islands), FR (France), GA (Gabon), GB (United Kingdom), GD (Grenada), GE (Georgia), GF (French Guiana), GG (Guernsey), GH (Ghana), GI (Gibraltar), GL (Greenland), GM (Gambia), GN (Guinea), GP (Guadeloupe), GQ (Equatorial Guinea), GR (Greece), GS (South Georgia and the South Sandwich Islands), GT (Guatemala), GU (Guam), GW (Guinea-Bissau), GY (Guyana), HK (Hong Kong), HM (Heard Island and McDonald Islands), HN (Honduras), HR (Croatia), HT (Haiti), HU (Hungary), ID (Indonesia), IE (Ireland), IL (Israel), IN (India), IQ (Iraq), IR (Iran), IS (Iceland), IT (Italy), JE (Jersey), JM (Jamaica), JO (Jordan), JP (Japan), KE (Kenya), KG (Kyrgyzstan), KH (Cambodia), KI (Kiribati), KM (Comoros), KN (Saint Kitts and Nevis), KP (North Korea), KR (Korea (South)), KW (Kuwait), KY (Cayman Islands), KZ (Kazakhstan), LA (Laos), LB (Lebanon), LC (Saint Lucia), LI (Liechtenstein), LK (Sri Lanka), LR (Liberia), LS (Lesotho), LT (Lithuania), LU (Luxembourg), LV (Latvia), LY (Libya), MA (Morocco), MC (Monaco), MD (Moldova), ME (Montenegro), MF (Saint Martin), MG (Madagascar), MH (Marshall Islands), MK (Macedonia), ML (Mali), MM (Myanmar), MN (Mongolia), MO (Macau), MP (Northern Mariana Islands), MQ (Martinique), MR (Mauritania), MS (Montserrat), MT (Malta), MU (Mauritius), MV (Maldives), MW (Malawi), MX (Mexico), MY (Malaysia), MZ (Mozambique), NA (Namibia), NC (New Caledonia), NE (Niger), NF (Norfolk Island), NG (Nigeria), NI (Nicaragua), NL (Netherlands), NO (Norway), NP (Nepal), NR (Nauru), NU (Niue), NZ (New Zealand), OM (Oman), PA (Panama), PE (Peru), PF (French Polynesia), PG (Papua New Guinea), PH (Philippines), PK (Pakistan), PL (Poland), PM (Saint Pierre and Miquelon), PN (Pitcairn Islands), PR (Puerto Rico), PS (Palestine), PT (Portugal), PW (Palau), PY (Paraguay), QA (Qatar), RE (Reunion), RO (Romania), RS (Serbia), RU (Russia), RW (Rwanda), SA (Saudi Arabia), SB (Solomon Islands), SC (Seychelles), SD (Sudan), SE (Sweden), SG (Singapore), SH (Saint Helena), SI (Slovenia), SJ (Svalbard and Jan Mayen), SK (Slovakia), SL (Sierra Leone), SM (San Marino), SN (Senegal), SO (Somalia), SR (Suriname), ST (Sao Tome and Principe), SV (El Salvador), SX (Sint Maarten), SY (Syria), TC (Turks and Caicos Islands), TD (Chad), TF (French Southern Territories), TG (Togo), TH (Thailand), TJ (Tajikistan), TK (Tokelau), TL (Timor-Leste), TM (Turkmenistan), TN (Tunisia), TO (Tonga), TR (Turkey), TT (Trinidad and Tobago), TV (Tuvalu), TW (Taiwan), TZ (Tanzania), UA (Ukraine), UG (Uganda), US (United States), UY (Uruguay), UZ (Uzbekistan), VC (Saint Vincent and the Grenadines), VE (Venezuela), VN (Vietnam), VU (Vanuatu), WF (Wallis and Futuna), WS (Samoa), YE (Yemen), YT (Mayotte), ZA (South Africa), ZM (Zambia), ZW (Zimbabwe). For the field 'locations.subregion_code' use only the predefined values: ANZ (Australia and New Zealand), CAM (Central America), CAS (Central Asia), EAS (Eastern Asia), EEU (Eastern Europe), LATAM (Latin America and the Caribbean), MEAS (Middle East Asia), MEL (Melanesia), MIC (Micronesia), NAF (North Africa), NAM (North America), NEU (Northern Europe), POL (Polynesia), SAM (South America), SAS (South Asia), SEAS (Southeast Asia), SEU (Southern Europe), SSAF (Sub-Saharan Africa), WAS (Western Asia), WEU (Western Europe). For the field 'locations.region_code' use only the predefined values: AF (Africa), AM (Americas), AN (Antarctica), AS (Asia), EU (Europe), OC (Oceania), WW (Worldwide)."
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
        num_train_epochs=3,  # Set this for 1 full training run.
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
