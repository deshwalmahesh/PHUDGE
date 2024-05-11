#For Multi GPU, either run it like:
# torchrun --nproc_per_node=4 --master_port=6543 ./train_phi-2-reg.py
# accelerate launch ./train_phi-3-reg_classif.py --deepspeed "auto"
# accelerate launch --config_file=./scripts/accelerate_default_config.yaml ./scripts/train_phi-3-reg_classif.py


import torch
from datasets import load_dataset, load_from_disk, load_metric
from datasets import Dataset
from transformers import  (AutoTokenizer, EarlyStoppingCallback, DataCollatorWithPadding, Phi3ForSequenceClassification, BitsAndBytesConfig,
                           TrainingArguments, Trainer)

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import numpy as np
import json
import pandas as pd
import time, random, os, wandb, warnings
from datetime import datetime
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union

from training_helpers import *


datestring = datetime.now().strftime("%d_%m_%Y_%H_%M")  # Date to the desired string format

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


wandb.login(key = "YOUR_KEY") # 
os.environ["WANDB_PROJECT"] = "PHUDGE-Classif"  # name your W&B project 

    
    
# ----------------- TRAIN -------------------
def train(BASE_PATH, num_classes, model_name, EPOCHS, BATCH, GRAD_ACC = 1, 
          use_peft = True, lora_rank = 128,  qlora = False, attn_implementation = None,  DEBUG = True):
    
    accelerator =  Accelerator()
    
     # ----- DATA -----
    
    tokenized_train_dataset = load_from_disk(os.path.join(BASE_PATH, "tokenized_train_dataset"))
    tokenized_val_dataset = load_from_disk(os.path.join(BASE_PATH, "tokenized_val_dataset"))
    
    assert (not "labels" in tokenized_train_dataset.features), "'labels' can't be there for Classification. This looks like a SFT dataset"

    if DEBUG:
        tokenized_train_dataset = tokenized_train_dataset.select(np.random.choice(range(len(tokenized_train_dataset)-1), 256))
        tokenized_val_dataset = tokenized_val_dataset.select(np.random.choice(range(len(tokenized_val_dataset)-1), 64))
    
    
    # --------------- PARAMS ---------------------
    # 1824 # Look at the train data input_ids length distribution
    
    steps_per_epoch = len(tokenized_train_dataset)//(accelerator.num_processes * BATCH * GRAD_ACC) # GPU * BATCH * GRAD_ACC
    EPOCHS = EPOCHS if not DEBUG else 2
    EVAL_STEPS = (steps_per_epoch // 2) if not DEBUG else 2 # Eval twice per epoch
    LOG_STEPS = 50 if not DEBUG else 2
    PATIENCE = 4 # 5 steps == 1.5 epochs
    
    SEED = 13

    seed_everything(seed = SEED)
    
    metrics = CustomMetrics(num_classes = num_classes) 
    
    # --------- TOKENIZER -----------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False, trust_remote_code = True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_id = tokenizer.eos_token_id

    
    bnb_config = None
    
    if qlora:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type='nf4',
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        # bnb_4bit_compute_dtype="float16",
                                        bnb_4bit_use_double_quant=False)

 
    # --------- MODEL -----------------
    id2label = {0:1, 1:2, 2:3, 3:4, 4:5}
    label2id = {5:4, 4:3, 3:2, 2:1, 1:0}
        
    model = Phi3ForSequenceClassification.from_pretrained(model_name, trust_remote_code = True, quantization_config = bnb_config,
                                                                 attn_implementation = attn_implementation, torch_dtype=torch.bfloat16, 
                                                                 num_labels = num_classes,
                                                                id2label = id2label, label2id = label2id, 
                                                                )
     
    if model.config.eos_token_id is None: model.config.eos_token_id = tokenizer.eos_token_id
    if  model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.pad_token_id
    
    model.gradient_checkpointing_enable() #gradient checkpointing to save memory

    if use_peft:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # Freeze base model layers and cast layernorm in fp32

        lora_config = LoraConfig(
            r = lora_rank,
            lora_alpha= lora_rank*2,
            target_modules=["o_proj","qkv_proj","gate_up_proj","down_proj"],
            bias="none",
            lora_dropout=0.1,
            task_type="SEQ_CLS")


        model = get_peft_model(model, lora_config) # LORA

    
    BASE_PATH = os.path.join(BASE_PATH, ("debug" if DEBUG else datestring)) # save data and results there based on date
    
    if accelerator.is_main_process:
        MAX_LENGTH = len(tokenized_train_dataset[0]["input_ids"])
                         
        wandb.init(project = os.environ["WANDB_PROJECT"],
                   config = {"DEBUG":DEBUG,"SEED":SEED, "MAX_LENGTH":MAX_LENGTH, 
                             "BASE_PATH":BASE_PATH, "attn_type":attn_implementation, 
                              "qlora":qlora, "use_peft":use_peft, "lora_rank": lora_rank,
                            "random_example": np.random.choice(tokenized_train_dataset["text"]),
                            })
        
        
        print_trainable_parameters(model)
        
        print("Data Length: ", len(tokenized_train_dataset), "Steps Per Epoch (Manual): ",steps_per_epoch)
        print("Sequence Length: ", MAX_LENGTH)
        print(np.random.choice(tokenized_train_dataset["text"]))
        
        print(model)
        
    # ------------- TRAINING -------------------------
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["text"])
    
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_PATH,'results'), 
        overwrite_output_dir=True, # Overwrite the content of the output directory
        per_device_train_batch_size= BATCH,  
        per_device_eval_batch_size= BATCH, 
        dataloader_num_workers = accelerator.num_processes,
        
        warmup_ratio = 0.05, 
        
        max_steps = -1,
        num_train_epochs = EPOCHS,  
        learning_rate=2e-5, 
        weight_decay=0.005, 
        optim =  "adamw_torch", #"adamw_apex_fused",#"adamw_torch", #Keep the optimizer state and quantize it
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
        save_total_limit = PATIENCE+1, # Eval thrice per epoch it means keep atleast 3 epochs checkpoints
        evaluation_strategy="steps",
        eval_steps = EVAL_STEPS,
        load_best_model_at_end = True, # Load the best model at the end of training
        report_to = 'wandb',
        neftune_noise_alpha = 7)
    
    
    class CustomLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            
            # Get model's predictions
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Compute custom loss
            loss_fct =  CustomEMDLoss(self.model.config.num_labels, device = model.device, with_sanity_check = True,
                                      p = 2,  # Not used when squared form is True
                                      alpha = 2, # Not used when squared is True
                                      squared = True, 
                                      summed = False)
            
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    
    trainer = CustomLossTrainer(
        model = model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics = CustomMetrics(num_classes).compute_metrics_for_scores ,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = PATIENCE, early_stopping_threshold = 0.0005)]
    )

    model.config.use_cache = False  # Disable cache to prevent warning, renable for inference

    trainer.train()  # Start training
 
    trainer.save_state()
    trainer.save_model(os.path.join(BASE_PATH, "results/final_model"))
    model.save_pretrained(os.path.join(BASE_PATH, "results/final_lora"))



if __name__ == "__main__":
    train(BASE_PATH = "PATH_TO_YOUR/PHUDGE/classif/", num_classes = 5,
        model_name = "microsoft/Phi-3-mini-4k-instruct", EPOCHS = 5, BATCH = 18, GRAD_ACC = 1,
          use_peft = True, qlora = False, lora_rank = 128, 
          attn_implementation = "flash_attention_2",
          DEBUG = False)

