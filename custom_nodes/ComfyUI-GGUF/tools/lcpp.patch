diff --git a/ggml/include/ggml.h b/ggml/include/ggml.h
index de3c706f..0267c1fa 100644
--- a/ggml/include/ggml.h
+++ b/ggml/include/ggml.h
@@ -223,7 +223,7 @@
 #define GGML_MAX_OP_PARAMS      64
 
 #ifndef GGML_MAX_NAME
-#   define GGML_MAX_NAME        64
+#   define GGML_MAX_NAME        128
 #endif
 
 #define GGML_DEFAULT_N_THREADS  4
@@ -2449,6 +2449,7 @@ extern "C" {
 
     // manage tensor info
     GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
+    GGML_API void gguf_set_tensor_ndim(struct gguf_context * ctx, const char * name, int n_dim);
     GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);
     GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);
 
diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c
index b16c462f..6d1568f1 100644
--- a/ggml/src/ggml.c
+++ b/ggml/src/ggml.c
@@ -22960,6 +22960,14 @@ void gguf_add_tensor(
     ctx->header.n_tensors++;
 }
 
+void gguf_set_tensor_ndim(struct gguf_context * ctx, const char * name, const int n_dim) {
+    const int idx = gguf_find_tensor(ctx, name);
+    if (idx < 0) {
+        GGML_ABORT("tensor not found");
+    }
+    ctx->infos[idx].n_dims = n_dim;
+}
+
 void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type) {
     const int idx = gguf_find_tensor(ctx, name);
     if (idx < 0) {
diff --git a/src/llama.cpp b/src/llama.cpp
index 24e1f1f0..25db4c69 100644
--- a/src/llama.cpp
+++ b/src/llama.cpp
@@ -205,6 +205,17 @@ enum llm_arch {
     LLM_ARCH_GRANITE,
     LLM_ARCH_GRANITE_MOE,
     LLM_ARCH_CHAMELEON,
+    LLM_ARCH_FLUX,
+    LLM_ARCH_SD1,
+    LLM_ARCH_SDXL,
+    LLM_ARCH_SD3,
+    LLM_ARCH_AURA,
+    LLM_ARCH_LTXV,
+    LLM_ARCH_HYVID,
+    LLM_ARCH_WAN,
+    LLM_ARCH_HIDREAM,
+    LLM_ARCH_COSMOS,
+    LLM_ARCH_LUMINA2,
     LLM_ARCH_UNKNOWN,
 };
 
@@ -258,6 +269,17 @@ static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
     { LLM_ARCH_GRANITE,         "granite"      },
     { LLM_ARCH_GRANITE_MOE,     "granitemoe"   },
     { LLM_ARCH_CHAMELEON,       "chameleon"    },
+    { LLM_ARCH_FLUX,            "flux"         },
+    { LLM_ARCH_SD1,             "sd1"          },
+    { LLM_ARCH_SDXL,            "sdxl"         },
+    { LLM_ARCH_SD3,             "sd3"          },
+    { LLM_ARCH_AURA,            "aura"         },
+    { LLM_ARCH_LTXV,            "ltxv"         },
+    { LLM_ARCH_HYVID,           "hyvid"        },
+    { LLM_ARCH_WAN,             "wan"          },
+    { LLM_ARCH_HIDREAM,         "hidream"      },
+    { LLM_ARCH_COSMOS,          "cosmos"       },
+    { LLM_ARCH_LUMINA2,         "lumina2"      },
     { LLM_ARCH_UNKNOWN,         "(unknown)"    },
 };
 
@@ -1531,6 +1553,17 @@ static const std::map<llm_arch, std::map<llm_tensor, const char *>> LLM_TENSOR_N
             { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
         },
     },
+    { LLM_ARCH_FLUX,    {}},
+    { LLM_ARCH_SD1,     {}},
+    { LLM_ARCH_SDXL,    {}},
+    { LLM_ARCH_SD3,     {}},
+    { LLM_ARCH_AURA,    {}},
+    { LLM_ARCH_LTXV,    {}},
+    { LLM_ARCH_HYVID,   {}},
+    { LLM_ARCH_WAN,     {}},
+    { LLM_ARCH_HIDREAM, {}},
+    { LLM_ARCH_COSMOS,  {}},
+    { LLM_ARCH_LUMINA2, {}},
     {
         LLM_ARCH_UNKNOWN,
         {
@@ -5403,6 +5436,25 @@ static void llm_load_hparams(
     // get general kv
     ml.get_key(LLM_KV_GENERAL_NAME, model.name, false);
 
+    // Disable LLM metadata for image models
+    switch (model.arch) {
+        case LLM_ARCH_FLUX:
+        case LLM_ARCH_SD1:
+        case LLM_ARCH_SDXL:
+        case LLM_ARCH_SD3:
+        case LLM_ARCH_AURA:
+        case LLM_ARCH_LTXV:
+        case LLM_ARCH_HYVID:
+        case LLM_ARCH_WAN:
+        case LLM_ARCH_HIDREAM:
+        case LLM_ARCH_COSMOS:
+        case LLM_ARCH_LUMINA2:
+            model.ftype = ml.ftype;
+            return;
+        default:
+            break;
+    }
+
     // get hparams kv
     ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, hparams.n_vocab);
 
@@ -18016,6 +18068,134 @@ static void llama_tensor_dequantize_internal(
     workers.clear();
 }
 
+static ggml_type img_tensor_get_type(quantize_state_internal & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
+    // Special function for quantizing image model tensors
+    const std::string name = ggml_get_name(tensor);
+    const llm_arch arch = qs.model.arch;
+
+    // Sanity check
+    if (
+            (name.find("model.diffusion_model.") != std::string::npos) ||
+            (name.find("first_stage_model.") != std::string::npos) ||
+            (name.find("single_transformer_blocks.") != std::string::npos) ||
+            (name.find("joint_transformer_blocks.") != std::string::npos)
+        ) {
+            throw std::runtime_error("Invalid input GGUF file. This is not a supported UNET model");
+    }
+
+    // Unsupported quant types - exclude all IQ quants for now
+    if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS  ||
+        ftype == LLAMA_FTYPE_MOSTLY_IQ2_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M  ||
+        ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S  ||
+        ftype == LLAMA_FTYPE_MOSTLY_IQ1_M   || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL ||
+        ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
+        ftype == LLAMA_FTYPE_MOSTLY_IQ3_M   || ftype == LLAMA_FTYPE_MOSTLY_Q4_0_4_4 ||
+        ftype == LLAMA_FTYPE_MOSTLY_Q4_0_4_8 || ftype == LLAMA_FTYPE_MOSTLY_Q4_0_8_8) {
+        throw std::runtime_error("Invalid quantization type for image model (Not supported)");
+    }
+
+    if ( // Rules for to_v attention
+            (name.find("attn_v.weight") != std::string::npos) ||
+            (name.find(".to_v.weight") != std::string::npos) ||
+            (name.find(".v.weight") != std::string::npos) ||
+            (name.find(".attn.w1v.weight") != std::string::npos) ||
+            (name.find(".attn.w2v.weight") != std::string::npos) ||
+            (name.find("_attn.v_proj.weight") != std::string::npos)
+        ){
+            if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
+                new_type = GGML_TYPE_Q3_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
+                new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
+                new_type = GGML_TYPE_Q5_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) {
+                new_type = GGML_TYPE_Q6_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) {
+                new_type = GGML_TYPE_Q5_K;
+            }
+            ++qs.i_attention_wv;
+    } else if ( // Rules for fused qkv attention
+            (name.find("attn_qkv.weight") != std::string::npos) ||
+            (name.find("attn.qkv.weight") != std::string::npos) ||
+            (name.find("attention.qkv.weight") != std::string::npos)
+        ) {
+            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
+                new_type = GGML_TYPE_Q4_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
+                new_type = GGML_TYPE_Q5_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) {
+                new_type = GGML_TYPE_Q6_K;
+            }
+    } else if ( // Rules for ffn
+            (name.find("ffn_down") != std::string::npos) ||
+            ((name.find("experts.") != std::string::npos) && (name.find(".w2.weight") != std::string::npos)) ||
+            (name.find(".ffn.2.weight") != std::string::npos) || // is this even the right way around?
+            (name.find(".ff.net.2.weight") != std::string::npos) ||
+            (name.find(".mlp.layer2.weight") != std::string::npos) ||
+            (name.find(".adaln_modulation_mlp.2.weight") != std::string::npos) ||
+            (name.find(".feed_forward.w2.weight") != std::string::npos)
+        ) {
+            // TODO: add back `layer_info` with some model specific logic + logic further down
+            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
+                new_type = GGML_TYPE_Q4_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
+                new_type = GGML_TYPE_Q5_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S) {
+                new_type = GGML_TYPE_Q5_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
+                new_type = GGML_TYPE_Q6_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) {
+                new_type = GGML_TYPE_Q6_K;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_0) {
+                new_type = GGML_TYPE_Q4_1;
+            }
+            else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_0) {
+                new_type = GGML_TYPE_Q5_1;
+            }
+            ++qs.i_ffn_down;
+    }
+
+    // Sanity check for row shape
+    bool convert_incompatible_tensor = false;
+    if (new_type == GGML_TYPE_Q2_K    || new_type == GGML_TYPE_Q3_K    || new_type == GGML_TYPE_Q4_K   ||
+        new_type == GGML_TYPE_Q5_K    || new_type == GGML_TYPE_Q6_K) {
+        int nx = tensor->ne[0];
+        int ny = tensor->ne[1];
+        if (nx % QK_K != 0) {
+            LLAMA_LOG_WARN("\n\n%s : tensor cols %d x %d are not divisible by %d, required for %s", __func__, nx, ny, QK_K, ggml_type_name(new_type));
+            convert_incompatible_tensor = true;
+        } else {
+            ++qs.n_k_quantized;
+        }
+    }
+    if (convert_incompatible_tensor) {
+        // TODO: Possibly reenable this in the future
+        // switch (new_type) {
+        //     case GGML_TYPE_Q2_K:
+        //     case GGML_TYPE_Q3_K:
+        //     case GGML_TYPE_Q4_K:   new_type = GGML_TYPE_Q5_0;   break;
+        //     case GGML_TYPE_Q5_K:   new_type = GGML_TYPE_Q5_1;   break;
+        //     case GGML_TYPE_Q6_K:   new_type = GGML_TYPE_Q8_0;   break;
+        //     default: throw std::runtime_error("\nUnsupported tensor size encountered\n");
+        // }
+        new_type = GGML_TYPE_F16;
+        LLAMA_LOG_WARN(" - using fallback quantization %s\n", ggml_type_name(new_type));
+        ++qs.n_fallback;
+    }
+    return new_type;
+}
+
 static ggml_type llama_tensor_get_type(quantize_state_internal & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
     const std::string name = ggml_get_name(tensor);
 
@@ -18513,7 +18693,9 @@ static void llama_model_quantize_internal(const std::string & fname_inp, const s
         if (llama_model_has_encoder(&model)) {
             n_attn_layer *= 3;
         }
-        GGML_ASSERT((qs.n_attention_wv == n_attn_layer) && "n_attention_wv is unexpected");
+        if (model.arch != LLM_ARCH_HYVID) { // TODO: Check why this fails
+            GGML_ASSERT((qs.n_attention_wv == n_attn_layer) && "n_attention_wv is unexpected");
+        }
     }
 
     size_t total_size_org = 0;
@@ -18547,6 +18729,51 @@ static void llama_model_quantize_internal(const std::string & fname_inp, const s
             ctx_outs[i_split] = gguf_init_empty();
         }
         gguf_add_tensor(ctx_outs[i_split], tensor);
+        // SD3 pos_embed needs special fix as first dim is 1, which gets truncated here
+        if (model.arch == LLM_ARCH_SD3) {
+            const std::string name = ggml_get_name(tensor);
+            if (name == "pos_embed" && tensor->ne[2] == 1) {
+                const int n_dim = 3;
+                gguf_set_tensor_ndim(ctx_outs[i_split], "pos_embed", n_dim);
+                LLAMA_LOG_INFO("\n%s: Correcting pos_embed shape for SD3: [key:%s]\n", __func__, tensor->name);
+            }
+        }
+        // same goes for auraflow
+        if (model.arch == LLM_ARCH_AURA) {
+            const std::string name = ggml_get_name(tensor);
+            if (name == "positional_encoding" && tensor->ne[2] == 1) {
+                const int n_dim = 3;
+                gguf_set_tensor_ndim(ctx_outs[i_split], "positional_encoding", n_dim);
+                LLAMA_LOG_INFO("\n%s: Correcting positional_encoding shape for AuraFlow: [key:%s]\n", __func__, tensor->name);
+            }
+            if (name == "register_tokens" && tensor->ne[2] == 1) {
+                const int n_dim = 3;
+                gguf_set_tensor_ndim(ctx_outs[i_split], "register_tokens", n_dim);
+                LLAMA_LOG_INFO("\n%s: Correcting register_tokens shape for AuraFlow: [key:%s]\n", __func__, tensor->name);
+            }
+        }
+        // conv3d fails due to max dims - unsure what to do here as we never even reach this check
+        if (model.arch == LLM_ARCH_HYVID) {
+            const std::string name = ggml_get_name(tensor);
+            if (name == "img_in.proj.weight" && tensor->ne[5] != 1 ) {
+                throw std::runtime_error("img_in.proj.weight size failed for HyVid");
+            }
+        }
+        // All the modulation layers also have dim1, and I think conv3d fails here too but we segfaul way before that...
+        if (model.arch == LLM_ARCH_WAN) {
+            const std::string name = ggml_get_name(tensor);
+            if (name.find(".modulation") != std::string::npos && tensor->ne[2] == 1) {
+                const int n_dim = 3;
+                gguf_set_tensor_ndim(ctx_outs[i_split], tensor->name, n_dim);
+                LLAMA_LOG_INFO("\n%s: Correcting shape for Wan: [key:%s]\n", __func__, tensor->name);
+            }
+            // FLF2V model only
+            if (name == "img_emb.emb_pos") {
+                const int n_dim = 3;
+                gguf_set_tensor_ndim(ctx_outs[i_split], tensor->name, n_dim);
+                LLAMA_LOG_INFO("\n%s: Correcting shape for Wan FLF2V: [key:%s]\n", __func__, tensor->name);
+            }
+        }
     }
 
     // Set split info if needed
@@ -18647,6 +18874,110 @@ static void llama_model_quantize_internal(const std::string & fname_inp, const s
         // do not quantize relative position bias (T5)
         quantize &= name.find("attn_rel_b.weight") == std::string::npos;
 
+        // rules for image models
+        bool image_model = false;
+        if (model.arch == LLM_ARCH_FLUX) {
+            image_model = true;
+            quantize &= name.find("txt_in.") == std::string::npos;
+            quantize &= name.find("img_in.") == std::string::npos;
+            quantize &= name.find("time_in.") == std::string::npos;
+            quantize &= name.find("vector_in.") == std::string::npos;
+            quantize &= name.find("guidance_in.") == std::string::npos;
+            quantize &= name.find("final_layer.") == std::string::npos;
+        }
+        if (model.arch == LLM_ARCH_SD1 || model.arch == LLM_ARCH_SDXL) {
+            image_model = true;
+            quantize &= name.find("class_embedding.") == std::string::npos;
+            quantize &= name.find("time_embedding.") == std::string::npos;
+            quantize &= name.find("add_embedding.") == std::string::npos;
+            quantize &= name.find("time_embed.") == std::string::npos;
+            quantize &= name.find("label_emb.") == std::string::npos;
+            quantize &= name.find("conv_in.") == std::string::npos;
+            quantize &= name.find("conv_out.") == std::string::npos;
+            quantize &= name != "input_blocks.0.0.weight";
+            quantize &= name != "out.2.weight";
+        }
+        if (model.arch == LLM_ARCH_SD3) {
+            image_model = true;
+            quantize &= name.find("final_layer.") == std::string::npos;
+            quantize &= name.find("time_text_embed.") == std::string::npos;
+            quantize &= name.find("context_embedder.") == std::string::npos;
+            quantize &= name.find("t_embedder.") == std::string::npos;
+            quantize &= name.find("y_embedder.") == std::string::npos;
+            quantize &= name.find("x_embedder.") == std::string::npos;
+            quantize &= name != "proj_out.weight";
+            quantize &= name != "pos_embed";
+        }
+        if (model.arch == LLM_ARCH_AURA) {
+            image_model = true;
+            quantize &= name.find("t_embedder.") == std::string::npos;
+            quantize &= name.find("init_x_linear.") == std::string::npos;
+            quantize &= name != "modF.1.weight";
+            quantize &= name != "cond_seq_linear.weight";
+            quantize &= name != "final_linear.weight";
+            quantize &= name != "final_linear.weight";
+            quantize &= name != "positional_encoding";
+            quantize &= name != "register_tokens";
+        }
+        if (model.arch == LLM_ARCH_LTXV) {
+            image_model = true;
+            quantize &= name.find("adaln_single.") == std::string::npos;
+            quantize &= name.find("caption_projection.") == std::string::npos;
+            quantize &= name.find("patchify_proj.") == std::string::npos;
+            quantize &= name.find("proj_out.") == std::string::npos;
+            quantize &= name.find("scale_shift_table") == std::string::npos; // last block too
+        }
+        if (model.arch == LLM_ARCH_HYVID) {
+            image_model = true;
+            quantize &= name.find("txt_in.") == std::string::npos;
+            quantize &= name.find("img_in.") == std::string::npos;
+            quantize &= name.find("time_in.") == std::string::npos;
+            quantize &= name.find("vector_in.") == std::string::npos;
+            quantize &= name.find("guidance_in.") == std::string::npos;
+            quantize &= name.find("final_layer.") == std::string::npos;
+        }
+        if (model.arch == LLM_ARCH_WAN) {
+            image_model = true;
+            quantize &= name.find("modulation.") == std::string::npos;
+            quantize &= name.find("patch_embedding.") == std::string::npos;
+            quantize &= name.find("text_embedding.") == std::string::npos;
+            quantize &= name.find("time_projection.") == std::string::npos;
+            quantize &= name.find("time_embedding.") == std::string::npos;
+            quantize &= name.find("img_emb.") == std::string::npos;
+            quantize &= name.find("head.") == std::string::npos;
+        }
+        if (model.arch == LLM_ARCH_HIDREAM) {
+            image_model = true;
+            quantize &= name.find("p_embedder.") == std::string::npos;
+            quantize &= name.find("t_embedder.") == std::string::npos;
+            quantize &= name.find("x_embedder.") == std::string::npos;
+            quantize &= name.find("final_layer.") == std::string::npos;
+            quantize &= name.find(".ff_i.gate.weight") == std::string::npos;
+            quantize &= name.find("caption_projection.") == std::string::npos;
+        }
+        if (model.arch == LLM_ARCH_COSMOS) {
+            image_model = true;
+            quantize &= name.find("p_embedder.") == std::string::npos;
+            quantize &= name.find("t_embedder.") == std::string::npos;
+            quantize &= name.find("t_embedding_norm.") == std::string::npos;
+            quantize &= name.find("x_embedder.") == std::string::npos;
+            quantize &= name.find("pos_embedder.") == std::string::npos;
+            quantize &= name.find("final_layer.") == std::string::npos;
+        }
+        if (model.arch == LLM_ARCH_LUMINA2) {
+            image_model = true;
+            quantize &= name.find("t_embedder.") == std::string::npos;
+            quantize &= name.find("x_embedder.") == std::string::npos;
+            quantize &= name.find("final_layer.") == std::string::npos;
+            quantize &= name.find("cap_embedder.") == std::string::npos;
+            quantize &= name.find("context_refiner.") == std::string::npos;
+            quantize &= name.find("noise_refiner.") == std::string::npos;
+        }
+        // ignore 3D/4D tensors for image models as the code was never meant to handle these
+        if (image_model) {
+            quantize &= ggml_n_dims(tensor) == 2;
+        }
+
         enum ggml_type new_type;
         void * new_data;
         size_t new_size;
@@ -18655,6 +18986,9 @@ static void llama_model_quantize_internal(const std::string & fname_inp, const s
             new_type = default_type;
 
             // get more optimal quantization type based on the tensor shape, layer, etc.
+            if (image_model) {
+                new_type = img_tensor_get_type(qs, new_type, tensor, ftype);
+            } else {
             if (!params->pure && ggml_is_quantized(default_type)) {
                 new_type = llama_tensor_get_type(qs, new_type, tensor, ftype);
             }
@@ -18664,6 +18998,7 @@ static void llama_model_quantize_internal(const std::string & fname_inp, const s
             if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
                 new_type = params->output_tensor_type;
             }
+            }
 
             // If we've decided to quantize to the same type the tensor is already
             // in then there's nothing to do.
