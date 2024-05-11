#For Multi GPU, either run it like:
# torchrun --nproc_per_node=4 --master_port=6543 ./train_sft_custom_data.py
# accelerate launch ./train_sft-custom_data.py --deepspeed "auto"

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
import transformers

from accelerate import Accelerator


from transformers import TrainingArguments, Trainer
from trl import  DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training

import numpy as np
import json
import pandas as pd
import time, random, os, wandb, warnings
from datetime import datetime
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Union

from datetime import datetime
datestring = datetime.now().strftime("%d_%m_%Y_%H")  # Date to the desired string format

from training_helpers import seed_everything, print_trainable_parameters, CustomDataCollatorForCompletionOnlyLM
from data_utils import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


wandb.login(key = "YOUR_API_KEY") # 
os.environ["WANDB_PROJECT"] = "PHUDGE-Causal"  # name your W&B project

    

def train(BASE_PATH, model_name, EPOCHS = 3, MAX_LENGTH = 1656, BATCH = 8, GRAD_ACC = 1, use_peft = True, lora_rank = 128, 
          qlora = False, completion_only = False, attn_implementation = None, PATIENCE = 5, DEBUG = True):
    """
    completion_only : whetehr to compute loss on Completion only for full generation. Data collator will be chosen based on that
    """
    
    accelerator =  Accelerator()
    
      # --------- TOKENIZER -----------------
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
     # ----- DATA -----
    prometheus = pd.read_parquet(os.path.join(BASE_PATH,"prometheus_with_contexts.parquet"))
    
    
    if DEBUG:
        tokenized_train_dataset =  TextDataset(prometheus["instruction"].values[:128], prometheus["output"].values[:128], tokenizer, MAX_LENGTH)
    else:
        tokenized_train_dataset = TextDataset(prometheus["instruction"].values, prometheus["output"].values, tokenizer, MAX_LENGTH)
    
    tokenized_val_dataset =  TextDataset(prometheus["instruction"].values[128:256], prometheus["output"].values[128:256], tokenizer, MAX_LENGTH)
       
    
    steps_per_epoch = len(tokenized_train_dataset)//(accelerator.num_processes * BATCH * GRAD_ACC) # GPU * BATCH * GRAD_ACC
    EPOCHS = EPOCHS if not DEBUG else 2
    EVAL_STEPS = (steps_per_epoch) if not DEBUG else 2 # Eval thrice per epoch
    LOG_STEPS = 100 if not DEBUG else 2

    
    BASE_PATH = os.path.join(BASE_PATH, ("debug" if DEBUG else datestring)) # save data and results there based on date
    
    
    SEED = 13

    seed_everything(seed = SEED)
    
     
    if completion_only: # Collator
        data_collator = CustomDataCollatorForCompletionOnlyLM(completion_only, tokenizer=tokenizer) if not DEBUG else DataCollatorForCompletionOnlyLM(completion_only, tokenizer=tokenizer)
    else: data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer , mlm = False)

    
    # -----------------------   QLoRA -------------------
    bnb_config = None
    
    if qlora:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type='nf4',
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        # bnb_4bit_compute_dtype="float16",
                                        bnb_4bit_use_double_quant=False)

 
    # --------- MODEL -----------------
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                 quantization_config=bnb_config,
                                                 attn_implementation=attn_implementation,
                                                 torch_dtype=torch.bfloat16)
    
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    
    model.gradient_checkpointing_enable() #gradient checkpointing to save memory

    if use_peft:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # Freeze base model layers and cast layernorm in fp32

        lora_config = LoraConfig(
            r = lora_rank,
            lora_alpha= lora_rank,
            modules_to_save = ["lm_head"],
            target_modules=["o_proj","qkv_proj","gate_up_proj","down_proj"], #print(model) will show the modules to use
            bias="none",
            lora_dropout=0.1,
            task_type="CAUSAL_LM")


        model = get_peft_model(model, lora_config) # LORA

    
    if accelerator.is_main_process:
        wandb.init(project = os.environ["WANDB_PROJECT"],
                   config = {"DEBUG":DEBUG,"SEED":SEED, "MAX_LENGTH":MAX_LENGTH, 
                             "BASE_PATH":BASE_PATH, "attn_type":attn_implementation, 
                              "qlora":qlora, "use_peft":use_peft, "lora_rank": lora_rank,
                            "random_example": tokenized_train_dataset.show_random_example(),
                             "collator": str(data_collator),
                            })
        
        print(tokenized_train_dataset.show_random_example())
        print_trainable_parameters(model)
        
        print(model)
        
    # ------------- TRAINING -------------------------
    
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_PATH,'results'), 
        overwrite_output_dir=True, # Overwrite the content of the output directory
        per_device_train_batch_size = BATCH,  
        per_device_eval_batch_size = BATCH, 
        dataloader_num_workers = accelerator.num_processes,
        
        # warmup_steps=50,  # Number of warmup steps
        warmup_ratio = 0.05, 
        
        max_steps = -1,
        num_train_epochs = EPOCHS,  
        learning_rate=2e-5, 
        weight_decay=0.05, 
        optim =  "adamw_torch", #"adamw_apex_fused",#"adamw_torch", # Loss might be oscillating with 8_bit
        lr_scheduler_type = "cosine",
        
        torch_compile = True,
        bf16=True,
        tf32 = True,
        gradient_accumulation_steps= GRAD_ACC,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},

        #For logging and saving
        logging_dir= os.path.join(BASE_PATH, 'logs'),
        logging_strategy="steps",
        logging_steps = LOG_STEPS,
        save_strategy="steps",
        save_steps = EVAL_STEPS,
        save_total_limit= PATIENCE + 2,  
        evaluation_strategy="steps",
        eval_steps = EVAL_STEPS,
        load_best_model_at_end = True, # Load the best model at the end of training
        report_to = 'wandb',
        neftune_noise_alpha = 7,
        
    )

    trainer = Trainer(
        model = model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator = data_collator,
        # compute_metrics = SFTClassificationMetric(tokenizer = tokenizer).sft_classification_score,
        # packing = False, # if True, you can't use DataCollatorForCompletionOnlyLM : use with SFTTrainer
        callbacks = [EarlyStoppingCallback(early_stopping_patience = PATIENCE)]

    )

    model.config.use_cache = False  # Disable cache to prevent warning, renable for inference

    trainer.train()  # Start training
 
    
    trainer.save_state()
    # trainer.save_model(os.path.join(BASE_PATH, "results/final_model"))
    # model.save_pretrained(os.path.join(BASE_PATH, "results/final_lora_adapters"))



if __name__ == "__main__":
    train(BASE_PATH = "PATH_TO_YOUR/PHUDGE/causal/", 
          model_name = "microsoft/Phi-3-mini-4k-instruct",
          EPOCHS = 3, MAX_LENGTH = 1656, BATCH = 8, GRAD_ACC = 1, 
          use_peft = True, lora_rank = 256, qlora = False, 
          completion_only = [32001], 
          attn_implementation = "flash_attention_2",
          DEBUG = False)

