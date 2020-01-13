# Module v1.0

## 使用方法
0. 声明调用  
    ```
    import Module.basic_module as bm
    import Module.abstract_network as an
    ```
    
1. 以Conv2d为例，将原有的nn.Conv2d替换为bm.Conv2d即可
    ```
    # nn.Conv2d(5, 4, 3, padding=1, bias=False)  
    bm.Conv2d(5, 4, 3, padding=1) # bias默认为False
    ```
    
2. 提供一个CBR整合模块BasicModule(Conv2d+BatchNorm2d+ReLU)
    ```
    bm.BasicModule(5,4,3,stride=1) # 自动计算padding
    ```

3. 网络继承从torch.nn改为bn
    ```
    # class DemoNet(nn.Module): pass
    class DemmoNet(an.AbstractNetwork):
        def __init__(self, layer_trunk, link_matrix):
            an.AbstractNetwork.__init__(self)
            self.layer_trunk = layer_trunk # 定义每层的性质
            self.link_matrix = link_matrix # 定义不同层之间的连接关系
        
        # def forward(self, *input):
        # 基本的backbone不再需要写forward函数
    ```

**更多内容可参考demo.py**

## 可用基本模块
* nn.Conv2d
* nn.BatchNorm2d

*激活函数可以直接调用nn,不影响剪枝
