# -*- coding: utf-8 -*-
import os

def deepspeed_config_from_args(args, global_batch_size):
    if args.use_zero_stage == 2:
        deepspeed_config = {
            "train_batch_size": global_batch_size,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accu_steps,
            "steps_per_print": args.log_every,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-08,
                    "weight_decay": args.weight_decay
                }
            },

            "zero_optimization": {
                "stage": 2,
                "reduce_scatter": False,
                "reduce_bucket_size": 1e9,
            },

            "gradient_clipping": 1.0,
            "prescale_gradients": True,

            "fp16": {
                "enabled": args.use_fp16,
                "loss_scale": 0,
                "loss_scale_window": 500,
                "hysteresis": 2,
                "min_loss_scale": 1e-3,
                "initial_scale_power": 15
            },

            "bf16": {
                "enabled": False
            },

            "wall_clock_breakdown": False
        }

    elif args.use_zero_stage == 3:
        deepspeed_config = {
            "train_batch_size": args.global_batch_size,
            # "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accu_steps,
            "steps_per_print": args.log_every,

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-08,
                    "weight_decay": args.weight_decay
                }
            },

            "zero_optimization": {
                "stage": 3,
                "allgather_partitions": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "contiguous_gradients": True,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_max_live_parameters" : 6e8,
                "reduce_bucket_size": 1.2e9,
                "sub_group_size": 1e9,
                "sub_group_buffer_num": 10,
                "pipeline_optimizer": True,
                "max_contigous_event_size": 0,
                "cache_sub_group_rate": 0.0,
                "prefetch_cache_sub_group_rate": 1.0,
                "max_contigous_params_size": -1,
                "max_param_reduce_events": 0,
                "stage3_param_persistence_threshold": 9e9,
                "is_communication_time_profiling": False,
                "save_large_model_multi_slice": True,
                "use_fused_op_with_grad_norm_overflow": False,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                }
            },

            "gradient_clipping": 1.0,
            "prescale_gradients": False,

            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 500,
                "hysteresis": 2,
                "min_loss_scale": 1,
                "initial_scale_power": 15
            },

            "bf16": {
                "enabled": False
            },

            "wall_clock_breakdown": False,
            "mem_chunk": {
                "default_chunk_size": 536870911,
                "use_fake_dist": False,
                "client": {
                    "mem_tracer": {
                        "use_async_mem_monitor": True,
                        "warmup_gpu_chunk_mem_ratio": 0.8,
                        "overall_gpu_mem_ratio": 0.8,
                        "overall_cpu_mem_ratio": 1.0,
                        "margin_use_ratio": 0.8,
                        "use_fake_dist": False
                    },
                    "opts": {
                        "with_mem_cache": True,
                        "with_async_move": True
                    }
                }
            }
        }
    else:
        raise ValueError
    return deepspeed_config


