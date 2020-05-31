归一化层，目前主要有这几个方法，Batch Normalization（2015年）、Layer Normalization（2016年）、Instance Normalization（2017年）
、Group Normalization（2018年）、Switchable Normalization（2018年）
————————————————  https://blog.csdn.net/liuxiao214/article/details/81037416

1. Batch Normalization : N H W  nn.BatchNorm2d
2. Layer Normalization : C H W  nn.LayerNorm
3. Instance Normalization : H W  nn.InstanceNorm2d
