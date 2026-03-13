
import logging
import time
import torch
from ml.get_local_job_config import load_local_training_config
from ml.cray_megatron.collectives.main_rank_only import log_if_main_rank
from tokenformer.transformers_tokenformer import TransformersTokenformerSurgeon
from peft import LoraConfig, get_peft_model

 
def log_param_gradients(model):
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    log_if_main_rank(f"Parameter summary: {trainable_count:,} trainable out of {total_count:,} total")


def create_llama_tokenformer_model(model, device, train_lm_head=None, adapter_type='lora'): 
    overall_start = time.time()

    if adapter_type == 'lora':
        job_config = load_local_training_config()

        log_if_main_rank("Inserting LoRA adapter...")
        step1_start = time.time()
        
        peft_config = LoraConfig(
            r=job_config.get("r", 8),
            lora_alpha=job_config.get("lora_alpha", 16),
            task_type="CAUSAL_LM",
            target_modules=job_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj" ]),
            lora_dropout=job_config.get("lora_dropout", 0.05)
        )

        model = get_peft_model(model, peft_config)
        
        step1_time = time.time() - step1_start
        log_if_main_rank(f"LoRA adapter module insertion completed: {step1_time:.2f}s ({step1_time/60:.1f} minutes)")
 
        log_if_main_rank("Logging parameter summary...")
        step2_start = time.time()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_layers = sum(1 for p in model.parameters() if p.requires_grad)
        total_layers = sum(1 for p in model.parameters())
        log_if_main_rank(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")
        log_if_main_rank(f"Trainable layers: {trainable_layers:,} || All layers: {total_layers:,}")
        step2_time = time.time() - step2_start
        log_if_main_rank(f"Parameter logging completed: {step2_time:.2f}s")

        total_time = time.time() - overall_start
        log_if_main_rank(f"Adding LoRA adapter total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        return model

    elif adapter_type == 'tokenformer':

        # Step 1: Insert adapter modules
        log_if_main_rank("Inserting TokenFormer adapter...")
        step2_start = time.time()
        tokenformer_model = TransformersTokenformerSurgeon(model, device).insert_adapter_modules()

        step2_time = time.time() - step2_start
        log_if_main_rank(f"TokenFormer adapter module insertion completed: {step2_time:.2f}s ({step2_time/60:.1f} minutes)")

        # Step 2: Count parameters for train_lm_head decision
        if train_lm_head is None:
            # Big models with more than 100M parameters don't need to train the lm_head
            # and getting the gradient scale right can be tricky.
            # Finally, the lm_head can be big and slow down adaptor loading in inference.
            log_if_main_rank("Counting parameters...")
            step3_start = time.time()
            param_count = count_parameters(tokenformer_model)
            step3_time = time.time() - step3_start
            train_lm_head = param_count < 100_000_000
            log_if_main_rank(f"Parameter counting completed: {step3_time:.2f}s, count={param_count:,}, train_lm_head={train_lm_head}")

        # Step 3: Freeze all parameters
        log_if_main_rank("Freezing all parameters...")
        step4_start = time.time()
        frozen_count = 0
        for param in tokenformer_model.parameters():
            param.requires_grad = False
            frozen_count += 1
        step4_time = time.time() - step4_start
        log_if_main_rank(f"Parameter freezing completed: {step4_time:.2f}s, frozen {frozen_count:,} parameters")

        # Step 4: Unfreeze tokenformer parameters
        log_if_main_rank("Unfreezing tokenformer parameters...")
        step5_start = time.time()
        unfrozen_count = 0
        for name, param in tokenformer_model.named_parameters():
            if any(module_name in name for module_name in ["tokenformer"]):
                param.requires_grad = True
                unfrozen_count += 1
        step5_time = time.time() - step5_start
        log_if_main_rank(f"Tokenformer parameter unfreezing completed: {step5_time:.2f}s, unfrozen {unfrozen_count:,} parameters")

        # Change made here - Counting paramteres after creating and unfreezing tokenformer model
        trainable = sum(p.numel() for p in tokenformer_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in tokenformer_model.parameters())
        log_if_main_rank(f"Tokenformer trainable: {trainable:,} / {total:,} = {100*trainable/total:.4f}%")

        # Step 5: Handle lm_head training
        # If lm_head should be included in training, set it as well.
        # In some models, lm_head is tied to embeddings and not included as a param.
        # So it's best to access it directly.
        step6_start = time.time()
        if train_lm_head:
            log_if_main_rank("Setting lm_head for training...")
            if hasattr(tokenformer_model, 'lm_head') and hasattr(tokenformer_model.lm_head, 'weight'):
                tokenformer_model.lm_head.weight.requires_grad = True
                log_if_main_rank("lm_head weight set to trainable")
            else:
                log_if_main_rank("WARNING: lm_head or lm_head.weight not found")
        step6_time = time.time() - step6_start
        log_if_main_rank(f"lm_head handling completed: {step6_time:.2f}s")

        # Step 6: Log parameter gradients (optional, can be expensive)
        log_if_main_rank("Logging parameter gradients...")
        step7_start = time.time()
        log_param_gradients(tokenformer_model)
        step7_time = time.time() - step7_start
        log_if_main_rank(f"Parameter gradient logging completed: {step7_time:.2f}s")

        total_time = time.time() - overall_start
        log_if_main_rank(f"Adding TokenFormer adapter total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        log_if_main_rank(f"Breakdown: adapter_insert={step2_time:.1f}s, param_count={step3_time:.1f}s, freeze={step4_time:.1f}s, unfreeze={step5_time:.1f}s, lm_head={step6_time:.1f}s, logging={step7_time:.1f}s")

        return tokenformer_model

# Define a function to count parameters
def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

