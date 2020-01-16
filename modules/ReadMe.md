# Module v0.3

## 使用方法
0. 声明调用  
    ```
    import module.basic_module as bm
    import module.abstract_network as an
    ```
    
1. 以Conv2d为例，将原有的nn.Conv2d替换为bm.Conv2d即可
    ```
    # nn.Conv2d(5, 4, 3, padding=1, bias=False)  
    bm.Conv2d(5, 4, 3, padding=1) # bias默认为False
    ```

2. 网络继承从torch.nn改为bn
    ```
    # class DemoNet(nn.Module): pass
    class DemoNet(an.AbstractNetwork): pass
    ```

*更多内容可参考demo.py*

## 目前可用的模块(2020.01.16)
* nn.Conv2d
* nn.BatchNorm2d
* 激活函数可以直接调用nn,不影响剪枝
