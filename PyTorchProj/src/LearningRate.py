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
    
    
4、学习率退火
https://blog.csdn.net/shanglianlm/article/details/85143614
    
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

使用实例:
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # cheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0, last_epoch=-1)
    
    for epoch in range(100):               
        train(...
             optimizer.zero_grad()
             optimizer.step()
             )
        validate(...)        
        scheduler.step()
        print('T is {}, and lr is {}'.format(scheduler.last_epoch, scheduler.get_lr[0]))
                                             # optim.param_groups[0]['lr']))

# T_max(int)- 一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。
# eta_min(float)- 最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0。



5、 Adam优化器，还需要学习率退火么？

https://www.cnblogs.com/wuliytTaotao/p/11101652.html
    Adam和学习率衰减（learning rate decay）
http://spytensor.com/index.php/archives/32/    
    pytorch 动态调整学习率

# 配合这个还蛮好的   
CLASS torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
      patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
patience指的是连续多少回没有达到、更新已经达到的min或max。
