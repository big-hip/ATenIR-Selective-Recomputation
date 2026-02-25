import torch
import torch.nn as nn # 记得导入 nn
from transformers import AutoTokenizer, AutoModelForCausalLM,GPT2LMHeadModel,GPT2Config
import os
import sys
import io
import shutil
from torch.fx.passes.graph_drawer import FxGraphDrawer
import Dist_IR
from Dist_IR import InferenceGraphCapture, logger, KVCacheAnalyzer
import multiprocessing as mp # 引入 multiprocessing
from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation

def save_inference_graph(graph_module, graph_name: str):
    """保存计算图为 MD 和 DOT 文件"""
    if not graph_module or not graph_module.graph.nodes:
        logger.info(f"Graph '{graph_name}' is empty, skipping save.")
        return

    output_dir = os.path.join('Dist_IR', graph_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # 清理旧文件
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 .md
    md_path = os.path.join(output_dir, 'Dist_IR_FW.md')
    with open(md_path, 'w') as f:
        original_stdout = sys.stdout
        with io.StringIO() as buf:
            sys.stdout = buf
            print(graph_module.graph)
            sys.stdout = original_stdout
            f.write(buf.getvalue())

    # 保存 .dot
    dot_path = os.path.join(output_dir, 'Dist_IR_FW.dot')
    try:
        g = FxGraphDrawer(graph_module, graph_name)
        with open(dot_path, 'w') as f:
            f.write(str(g.get_dot_graph()))
        logger.info(f"[Save] Saved {graph_name} to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save DOT for {graph_name}: {e}")
# --- 1. 新增 Wrapper ---
class GPT2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, past_key_values=None):
        # 强制 return_dict=False 以获得 Tuple 输出，这对 Graph Capture 更友好
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=False 
        )
        return outputs[0], outputs[1]
if __name__ == "__main__":
    # --- 关键：设置 spawn 启动方式 ---
    try:
        mp.set_start_method('spawn', force=True)
        print("[Main] Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"[Warn] Cannot set start method: {e}")
    # 通用设置  
    MODEL_NAME = "gpt2"
    gpu_index = 0
    DEVICE = "cpu"
    BATCH_SIZE = 4
    PREFILL_SEQ_LEN = 32 # 稍微调小一点方便调试
    KV_CACHE_SEQ_LEN = PREFILL_SEQ_LEN 


    try:
        # --- [核心修改] 手动构建 Tiny Config ---
        config = GPT2Config(
            n_layer=2,        # 层数减少到 2
            n_head=4,         # 头数减少到 4
            n_embd=128,       # 隐藏层维度减少到 128 (128 % 4 == 0)
            vocab_size=1000,  # 词表减小，减小 Embedding 算子大小
            n_positions=1024  # 最大长度
        )
        
        # 使用配置初始化模型（随机权重）
        raw_model = GPT2LMHeadModel(config).to(DEVICE)
        raw_model.eval()

        # 使用 Wrapper
        model = GPT2Wrapper(raw_model).to(DEVICE)
        
        print(f"   Config: Layers={config.n_layer}, Hidden={config.n_embd}, Heads={config.n_head}")

    except Exception as e:
        sys.exit(f"模型初始化失败: {e}")



    # print(f"使用的模型: {MODEL_NAME}, 设备: {DEVICE}")
    # raw_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    # raw_model.eval()
    # # 使用 Wrapper
    # model = GPT2Wrapper(raw_model).to(DEVICE)

    def create_dummy_kv_cache(config, batch_size, seq_len, device='cpu'):
        """创建符合 Llama 结构的伪造 KV Cache"""
        # Llama 2 可能使用 GQA，需要检查 num_key_value_heads
        num_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        num_layers = config.num_hidden_layers
        
        kv_cache = []
        for _ in range(num_layers):
            # Llama cache shape: [batch, n_kv_heads, seq_len, head_dim]
            k = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16)
            v = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16)
            kv_cache.append((k, v))
        return tuple(kv_cache)

    # 全局禁用梯度
    with torch.no_grad():
        
        # --- Prefill 捕获 ---
        print("\n" + "="*50)
        print("正在捕获 Prefill 阶段...")
        prefill_kwargs = { 
            'input_ids': torch.randint(0, raw_model.config.vocab_size, (BATCH_SIZE, PREFILL_SEQ_LEN), dtype=torch.long).to(DEVICE),
            'past_key_values': None, 
        }
        # 注意：InferenceGraphCapture 内部需要适配 Wrapper 的参数
        prefill_capture = InferenceGraphCapture(model, filename_prefix="prefill", **prefill_kwargs)
        compiled_model_prefill = prefill_capture.compile()
        compiled_model_prefill(**prefill_kwargs) # 触发 trace
        
        # --- Decode 捕获 ---
        print("\n" + "="*50)
        print("正在捕获 Decode 阶段...")
        dummy_kv = create_dummy_kv_cache(raw_model.config, BATCH_SIZE, KV_CACHE_SEQ_LEN, device=DEVICE)
        decode_kwargs = { 
            'input_ids': torch.randint(0, raw_model.config.vocab_size, (BATCH_SIZE, 1), dtype=torch.long).to(DEVICE),
            'past_key_values': dummy_kv, 
        }
        decode_capture = InferenceGraphCapture(model, filename_prefix="decode", **decode_kwargs)
        compiled_model_decode = decode_capture.compile()
        compiled_model_decode(**decode_kwargs) # 触发 trace


        # --- 定位 Pass ---
        # 对 Prefill
    if prefill_capture.FW_gm:
        Dist_IR.KVCacheAnalyzer(prefill_capture.FW_gm, "prefill", MODEL_NAME, raw_model.config)
        print("\n[Pos] Running Positioning on Prefill...")

        pos_prefill = Dist_IR.Pos(raw_model, prefill_capture.FW_gm, None, model_name='gpt2')
        pos_prefill.positioning_for_graph()
        pos_prefill.log_node_positioning_info()
        
        # 对 Decode 
        Dist_IR.KVCacheAnalyzer(prefill_capture.FW_gm, "decode", MODEL_NAME, raw_model.config)
        print("\n[Pos] Running Positioning on Decode...")
        pos_decode = Dist_IR.Pos(raw_model, decode_capture.FW_gm, None, model_name='gpt2')
        pos_decode.positioning_for_graph()
        
        # --- 混合并行 Pass ---
        print("\n[Hybrid] 应用并行策略...")
        
        # 1. Prefill Pass
        prefill_graph = Dist_IR.Hybrid_Parallel_pass(
            prefill_capture.FW_gm, 
            None, None, # 无 BW/OPT
            Global_batch_size=BATCH_SIZE,
            inference_stage='prefill'
        )
        
        # 2. Decode Pass (新增)
        decode_graph = Dist_IR.Hybrid_Parallel_pass(
            decode_capture.FW_gm, 
            None, None,
            Global_batch_size=BATCH_SIZE,
            inference_stage='decode'
        )
        
        # --- 保存处理后的图 ---
        print("\n[Save] 保存并行处理后的 ATen 图...")
        save_inference_graph(prefill_capture.FW_gm, "prefill_sharded")
        save_inference_graph(decode_capture.FW_gm, "decode_sharded")

        print("\nDone! 请检查输出目录。")   
        # from Performance_Eval.Fake_Runtime.PD_separation import PD_Performance_Eval

        # Eval = PD_Performance_Eval(prefill_capture.FW_gm, decode_capture.FW_gm, prefill_graph, decode_graph, PREFILL_SEQ_LEN, 2)
        # time = Eval.Evaluate()

        # print(f"最终结果: {time} us")
        from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation
        Eval = Performance_Evaluation(prefill_graph)
        time = Eval.Evaluate()
        print(f"最终结果: {time} us")
        