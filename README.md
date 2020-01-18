# IIV Pruning Tools v0.5

* Author: Blanc
* Tel: 13918670229
* E-mail: 1931604@tongji.edu.cn

## 使用方法
01. 声明调用  
    ```
    import pruning_tools as pt
    ```
11. 网络分析
    ```
    pt.analyze_network(
        network,                    # 需要分析的网络
        example_data,               # 测试网络的示例输入
        verbose=False,              # 是否显示分析结果
        for_pruning=True,           # 是否添加剪枝所需的属性和方法
        ) -> None
    ```
21. 一次性剪枝
    ```
    pt.one_shut_pruning(
        network,                    # 需要剪枝的网络
        example_data,               # 测试网络的示例输入
        method="minimum_weight",    # 剪枝方法
        pruning_rate=0.1,           # 目标剪枝率
        ) -> network                # 返回修剪后的网络
    ```
    一次成功的剪枝预计输出以下内容：
    >   Start one-shot pruning.  
        FLOPs before pruning is 593920, target is 475136.  
        Pruning Channel 11 in Layer .conv1, .layer1[1].conv2, .layer1[0].conv2,  (Score 0.2874)  
        Pruning Channel 42 in Layer .conv1, .layer1[1].conv2, .layer1[0].conv2,  (Score 0.2978)  
        ...  
        Pruning Channel 29 in Layer .conv1, .layer1[1].conv2, .layer1[0].conv2,  (Score 0.3408)  
        Successfully prune the network, the FLOPs now is 474112

22. 迭代式剪枝
    ```
    pt.iterative_pruning(
        network,                    # 需要剪枝的网络 
        example_data,               # 测试网络的示例输入
        func_train_one_epoch,       # 迭代使用的训练函数
        *training_args,             # 迭代使用的训练参数
        method="minimum_weight",    # 剪枝方法
        pruning_rate=0.1,           # 目标剪枝率
        pruning_interval=1,         # 大于1为两次剪枝间的训练轮数 小于1为两次训练间的剪枝轮数
        ) -> network                # 返回修剪后的网络
    ```
    一次成功的剪枝预计输出以下内容：
    >   Start iterative pruning.  
        FLOPs before pruning is 593920, target is 475136.  
        Epoch 1, Loss: 1716.7428, FLOPs: 593920  
        Pruning Channel 5 in Layer .conv1, .layer1[1].conv2, .layer1[0].conv2,  (Score 0.1760)  
        ...  
        
23. 自动剪枝(推荐)
    ```
    pt.automatic_pruning(
        network,                    # 需要剪枝的网络 
        example_data,               # 测试网络的示例输入
        func_valid,                 # 迭代使用的验证函数
        func_train_one_epoch,       # 迭代使用的训练函数
        *training_args,             # 迭代使用的训练参数
        method="minimum_weight",    # 剪枝方法
        epochs=10,                  # 自动剪枝迭代轮数
        ) -> network_backup         # 返回剪枝过程中最低Loss的网络
    ```
    一次成功的剪枝预计输出以下内容：
    >
