# Methods v0.3

## 使用方法
0. 声明调用并初始化剪枝工具  
    ```
    import method.abstract_method as pruning_methods
    
    pruning_tool = pruning_methods.PruningTool(
        input_channel=1,
        method="minimum_weight",
        pruning_rate=0.1)
    ```

*更多内容可参考demo.py*