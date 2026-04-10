import torch
import torch.nn as nn
import torch.optim as optim
import copy


def demo_model_deepcopy_effect():
    print("\n=== deepcopy 作用演示（模型对象/参数对象/参数存储）===")

    transformer = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=False),
    )

    # 仅做引用，不会复制任何参数或存储。
    alias_model = transformer
    clean_model = copy.deepcopy(transformer)

    orig_p = next(transformer.parameters())
    alias_p = next(alias_model.parameters())
    copy_p = next(clean_model.parameters())

    print("1) 模型对象 id")
    print(f"   transformer id: {id(transformer)}")
    print(f"   alias_model id: {id(alias_model)} (与 transformer 相同，说明只是别名)")
    print(f"   clean_model id: {id(clean_model)} (不同，说明是新对象)")

    print("2) 参数对象 id")
    print(f"   orig param id : {id(orig_p)}")
    print(f"   alias param id: {id(alias_p)} (与 orig 相同)")
    print(f"   copy param id : {id(copy_p)} (不同)")

    print("3) 参数底层存储地址")
    print(f"   orig data_ptr : {orig_p.data_ptr()}")
    print(f"   alias data_ptr: {alias_p.data_ptr()} (与 orig 相同)")
    print(f"   copy data_ptr : {copy_p.data_ptr()} (不同，说明权重存储也复制了)")

    with torch.no_grad():
        orig_p.add_(10.0)

    print("4) 修改原模型参数后，对比参数和是否受影响")
    print(f"   orig first elem : {orig_p.view(-1)[0].item():.6f}")
    print(f"   alias first elem: {alias_p.view(-1)[0].item():.6f} (会一起变化)")
    print(f"   copy first elem : {copy_p.view(-1)[0].item():.6f} (不变)")

    print("5) 结论")
    print("   clean_model = copy.deepcopy(transformer) 会复制模型结构和参数数据，")
    print("   得到彼此独立的参数对象与存储，常用于保留基线模型快照。")

def check_training_memory_usage():
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    if not torch.cuda.is_available():
        print("未检测到 GPU，无法演示显存动态变化。")
        return

    # 1. 初始状态
    base_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"1. 初始显存: {base_mem:.2f} MB")

    # 2. 构建模型并移动到 GPU
    # 两层 4096*4096，约 33.5M 个 float32 参数
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096)
    ).cuda()
    
    model_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"2. 模型加载后显存: {model_mem:.2f} MB (净增: {model_mem - base_mem:.2f} MB)")

    # 3. 创建优化器 (注意：Adam 会为每个参数存储 2 个状态量)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 4. 模拟一个 Batch 的数据
    # Batch size = 32
    input_data = torch.randn(32, 4096).cuda()
    target = torch.randn(32, 4096).cuda()
    print(f"3. 输入数据就绪后显存: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # 5. 前向传播 (Forward)
    output = model(input_data)
    loss = nn.MSELoss()(output, target)
    forward_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"4. 前向传播后 (激活值暂存): {forward_mem:.2f} MB")

    # 6. 反向传播 (Backward) - 显存高峰期
    loss.backward()
    backward_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"5. 反向传播后 (计算出梯度): {backward_mem:.2f} MB")

    # 7. 参数更新 (Step)
    optimizer.step()
    final_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"6. 优化器更新后 (Adam 状态初始化完成): {final_mem:.2f} MB")

    print("-" * 30)
    print(f"总结：从静态模型到真实训练，显存从 {model_mem:.2f} MB 飙升至 {final_mem:.2f} MB")
    print(f"增量主要来自：1.激活值(Forward) 2.梯度(Backward) 3.优化器矩(Optimizer States)")

demo_model_deepcopy_effect()
check_training_memory_usage()