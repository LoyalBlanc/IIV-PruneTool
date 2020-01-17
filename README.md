# IIV Pruning Tools v0.3

* Author: Blanc
* Tel: 13918670229
* E-mail: 1931604@tongji.edu.cn

## 使用方法
0. 声明调用  
    ```
    import module.basic_module as bm
    import module.abstract_network as an
    import method.abstract_method as pruning_methods
    ```
1. 构建网络
    pass



## Todo list:
### Version 0.4 
1. 修正automatic方法的训练bug
2. 测试现有剪枝方法在demo network上的效果
3. 将network_analysis函数从abstract_network中独立一个版本出来

### Version 0.5 (!!!)
1. 实现模块转移
2. Resnet18网络分析
    ```
    import torchvision.models as models
    resnet18 = models.resnet18()
    ```
3. 剪枝Resnet18

### Version 0.6
1. 使用Resnet50训练MNIST并剪枝
    ```
    resnext50_32x4d = models.resnext50_32x4d()
    ```
2. 完善注释和文档并发布