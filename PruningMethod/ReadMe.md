# Method v0.2

## 使用方法
0. 声明调用  
    ```
    import PruningMethod.methods as pm
    ```
    
1. 一次性剪枝  
    根据预设的剪枝方法和期望的剪枝率对网络进行一次性地修剪：
    ```
    pm.one_shot_prune(
        network,                    # 希望修剪的网络 推荐载入预训练参数
        example_data=None,          # 用于计算FLOPs 不提供时默认以32*32的图像计算
        method="minimum_weight",    # 剪枝方法 默认为最小参数修剪
        prune_ratio=0.1             # 期望剪枝率
        ) -> network                # 返回修剪后的网络
    ```
    一次成功的剪枝预期输出内容：
    >   Start one-shot pruning.  
        FLOPs before pruning is 600064, target is 540057.  
        Prune Channel 2 in Layer 12 (Weight 0.0106).  
        ...  
        Prune Channel 1 in Layer 29 (Weight 0.0160).  
        Successfully prune the network, the FLOPs now is 538624
    
2. 迭代式剪枝  
    每次剪枝后重新训练一轮网络直至达到期望的剪枝率：
    ```
    pm.iterative_prune(
        network,                    # 希望修剪的网络 推荐载入预训练参数
        dataset,                    # 训练所使用的数据集 推荐DataLoader格式
        example_data=None,          # 用于计算FLOPs 不提供时默认以32*32的图像计算
        method="minimum_weight",    # 剪枝方法 默认为最小参数修剪
        prune_ratio=0.1,            # 期望剪枝率
        criterion=nn.MSELoss(),     # 训练所使用的损失函数 剪枝的正则化会自动添加
        lr=1e-3                     # 训练所使用的学习率 优化方法为Adam
        ) -> network                # 返回修剪后的网络
    ```
    一次成功的剪枝预期输出内容：
    >   Start iterative pruning.  
        FLOPs before pruning is 600064, target is 540057.  
        Prune Channel 2 in Layer 12 (Weight 0.0106).  
        FLOPs: 593920,  Loss: 7.6135  
        ...  
        Prune Channel 1 in Layer 29 (Weight 0.0160).
        FLOPs: 538624,  Loss: 8.0119
        Successfully prune the network, the FLOPs now is 538624
        
3. 自动剪枝 
    每次剪枝后重新训练网络直至loss下降或超过预设训练轮数：
    ```
    pm.automotive_prune(
        network,                    # 希望修剪的网络 推荐载入预训练参数
        dataset,                    # 训练所使用的数据集 推荐DataLoader格式
        example_data=None,          # 用于计算FLOPs 不提供时默认以32*32的图像计算
        method="minimum_weight",    # 剪枝方法 默认为最小参数修剪
        prune_ratio=0.1,            # 期望剪枝率
        criterion=nn.MSELoss(),     # 训练所使用的损失函数 剪枝的正则化会自动添加
        lr=1e-3,                    # 训练所使用的学习率 优化方法为Adam
        epoch_limit=10,             # N轮未降低loss则停止剪枝
        step_loss_decay=0.5         # 每轮更新step loss的权重
        ) -> network_backup         # 返回最后一次修剪前的网络
    ```
    一次成功的剪枝预期输出内容：
    >   Start automotive pruning.  
        FLOPs before pruning is 600064, target is 360038.  
        Prune Channel 1 in Layer 3 (Weight 0.0299).  
        ...  
        Loss: 13.1803, Step Loss: 14.9309, new FLOPs: 581632  
        Loss: 15.3838, Step Loss: 14.9309  
        ...  
        Loss: 15.3847, Step Loss: 14.9309  
        Exceed the epoch limit and terminate pruning, the FLOPs now is 587776  
        Successfully prune the network, the FLOPs now is 587776  

*更多内容可参考demo.py*