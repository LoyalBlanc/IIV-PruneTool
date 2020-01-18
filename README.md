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
    根据预设的剪枝方法和目标剪枝率对网络进行一次性修剪。
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
        ...  
        Successfully prune the network, the FLOPs now is 474112

22. 迭代式剪枝  
    在一次性剪枝的基础上根据剪枝间隔添加Fine-tuning环节。  
    剪枝间隔(pruning_interval)：大于1为两次剪枝间的训练轮数，小于1为两次训练间的剪枝轮数。
    ```
    pt.iterative_pruning(
        network,                    # 需要剪枝的网络 
        example_data,               # 测试网络的示例输入
        func_train_one_epoch,       # 迭代使用的训练函数
        *training_args,             # 迭代使用的训练参数
        method="minimum_weight",    # 剪枝方法
        pruning_rate=0.1,           # 目标剪枝率
        pruning_interval=1,         # 剪枝间隔
        ) -> network                # 返回修剪后的网络
    ```
    一次成功的剪枝预计输出以下内容：
    >   Start iterative pruning.  
        FLOPs before pruning is 593920, target is 475136.  
        Epoch 1, Loss: 1716.7428, FLOPs: 593920  
        Pruning Channel 5 in Layer .conv1, .layer1[1].conv2, .layer1[0].conv2,  (Score 0.1760)  
        ...    
        Successfully prune the network, the FLOPs now is 455680
        
23. 自动剪枝(推荐)  
    根据提供的验证函数，在保证Accuracy的基础上尽可能地修剪网络。
    ```
    pt.automatic_pruning(
        network,                    # 需要剪枝的网络 
        example_data,               # 测试网络的示例输入
        func_valid,                 # 迭代使用的验证函数
        target_accuracy,            # 目标正确率
        func_train_one_epoch,       # 迭代使用的训练函数
        *training_args,             # 迭代使用的训练参数
        method="minimum_weight",    # 剪枝方法
        epochs=10,                  # 自动剪枝的最大迭代轮数
        ) -> network_backup         # 返回剪枝过程中最低Loss的网络
    ```
    一次成功的剪枝预计输出以下内容：
    >   Start automatic pruning.  
        Pruning Channel 177 in Layer .layer4[0].conv1,  (Score 0.0996)  
        Update network backup, FLOPs: 584752  
        Epoch [1/200], Loss: 27.8687, Accuracy: 98.99  
        Epoch [2/200], Loss: 25.5174, Accuracy: 99.17  
        Pruning Channel 154 in Layer .layer4[0].conv1,  (Score 0.0511)  
        ...  
        Successfully prune the network, the FLOPs now is 343524


*注：可参考demo.py在CIFAR10上对ResNet18进行的自动剪枝*

## 支持的模块(2020.01.18)
* nn.Conv2d
* nn.BatchNorm2d
* 任意不影响剪枝的模块，如各类激活函数