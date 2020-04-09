1、冻结层不参与训练方法：

######### 模型定义 #########
class MyModel(nn.Module):
    def __init__(self, feat_dim):   # input the dim of output fea-map of Resnet:
        super(MyModel, self).__init__()
        
        BackBone = models.resnet50(pretrained=True)
        
        add_block = []
        add_block += [nn.Linear(2048, 512)]
        add_block += [nn.LeakyReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_xavier)
 
        self.BackBone = BackBone
        self.add_block = add_block
 
 
    def forward(self, input):   # input is 2048!
 
        x = self.BackBone(input)
        x = self.add_block(x)
 
        return x
##############################
 
# 模型准备
model = MyModel()
 
# 优化、正则项、权重设置与冻结层
 
for param in model.parameters():
    param.requires_grad = False
for param in model.add_block.parameters():
    param.requires_grad = True
 
optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),  # 记住一定要加上filter()，不然会报错。 filter用法：https://www.runoob.com/python/python-func-filter.html
            lr=0.01,
            weight_decay=1e-5, momentum=0.9, nesterov=True)
 
 

2、各层采用不同学习率方法

######### 模型定义 #########
class MyModel(nn.Module):
    def __init__(self, feat_dim):   # input the dim of output fea-map of Resnet:
        super(MyModel, self).__init__()
        
        BackBone = models.resnet50(pretrained=True)
        
        add_block = []
        add_block += [nn.Linear(2048, 512)]
        add_block += [nn.LeakyReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_xavier)
 
        self.BackBone = BackBone
        self.add_block = add_block
 
 
    def forward(self, input):   # input is 2048!
 
        x = self.BackBone(input)
        x = self.add_block(x)
 
        return x
##############################
 
# 模型准备
model = MyModel()
 
# 不同层学习率设置
 
ignored_params = list(map(id, model.add_block.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
 
optimizer = optim.SGD(
             [
                {'params': base_params, 'lr': 0。01},
                {'params': model.add_block.parameters(), 'lr': 0.1},
                ]
            weight_decay=1e-5, momentum=0.9, nesterov=True)
 

3、调整学习率衰减。

方法一：使用torch.optim.lr_scheduler()函数：

####################
#  model structure
#-------------------
model = Mymodel()
if use_gpu:
    model = model.cuda()
 
####################
#        loss
#-------------------
criterion = nn.CrossEntropyLoss()
 
####################
#    optimizer
#-------------------
ignored_params = list(map(id, model.ViewModel.viewclassifier.parameters())) + list(map(id, model.Block.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.01},
        {'params': model.ViewModel.viewclassifier.parameters(), 'lr': 0.001},
        {'params': model.Block.parameters(), 'lr': 0.01}
    ], weight_decay=1e-3, momentum=0.9, nesterov=True)
 
 
####################
#**  Set lr_decay  **
#-------------------
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
 
scheduler.step()   # put it before model.train(True)
model.train(True)  # Set model to training mode
 
....
 

方法二：使用optimizer.param_groups方法。（好处：能分别设定不同层的衰减率！）

####################
#  model structure
#-------------------
model = Mymodel()
if use_gpu:
    model = model.cuda()
 
####################
#        loss
#-------------------
criterion = nn.CrossEntropyLoss()
 
####################
#    optimizer
#-------------------
ignored_params = list(map(id, model.ViewModel.viewclassifier.parameters())) + list(map(id, model.Block.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.01},
        {'params': model.ViewModel.viewclassifier.parameters(), 'lr': 0.001},
        {'params': model.Block.parameters(), 'lr': 0.03}],  
    weight_decay=1e-3, momentum=0.9, nesterov=True)
 
 
####################
#**  Set lr_decay  **
#-------------------
 
def adjust_lr(epoch):
    step_size = 60
    lr = args.lr * (0.1 ** (epoch // 30))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr')
 
    ######################################
    ###  optimizer.param_groups 类型与内容
    [
        { 'params': base_params, 'lr': 0.01, 'momentum': 0.9, 'dampening': 0,
        'weight_decay': 0.001, 'nesterov': True, 'initial_lr': 0.01 }, 
        { 'params': model.ViewModel.viewclassifier.parameters(), 'lr': 0.001, 
        'momentum': 0.9, 'dampening': 0, 'weight_decay': 0.001, 'nesterov': True, 
        'initial_lr': 0.001 },
        { 'params': model.Block.parameters(), 'lr': 0.03, 'momentum': 0.9, 
        'dampening': 0, 'weight_decay': 0.001, 'nesterov': True, 'initial_lr': 
        0.03 }
    ]
    ###  optimizer.param_groups 类型与内容
    ######################################
 
 
for epoch in range(start_epoch, args.epochs):
    adjust_lr(epoch)   # 每epoch更新一次。
    model.train(True)  # Set model to training mode
    ....
 
————————————————
版权声明：本文为CSDN博主「小小的行者」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/jdzwanghao/article/details/83239111
