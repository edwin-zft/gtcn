import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

# 数据传入
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))  # 求和 之后的大小是 N C W L
        return x.contiguous()  # 返回的X不改变之前的x

#最后的线性输出
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
# 图卷积
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]

        for a in support:

            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    """
    gcn_bool:使用图卷积
    addaptadj:使用自适应矩阵
    apinit:矩阵初始化
    in_dim:输入通道维度
    out_dim:输出通道维度  对应12个时间步
    residual_channels:残差网络通道
    dilation_channels:扩散卷积通道
    skip_channels:
    end_channels:

    """
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout  # drop
        self.blocks = blocks  #
        self.layers = layers  # 模型使用两层的因果卷积
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        # to expend channels -> residual_channels
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1  # 感受野

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:  # 图卷积
            if aptinit is None:  #
                if supports is None:  #
                    self.supports = []
                #生成参数 并且加入到parameter
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1



        # 4个块
        for b in range(blocks):
            additional_scope = kernel_size - 1  # padding
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
                    # gcn (32, 32, dropout, length of matrix)



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        """
        input：时间维度上填充一个维度 size = 64 2 325 13
        """
        in_len = input.size(3)  # 输入序列长度是时间序列长度
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))

        else:
            x = input

        x = self.start_conv(x)  # 64, 2, 325, 12 -> 64, 32, 325, 12
        skip = 0


        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers  4 * 2
        for i in range(self.blocks * self.layers):


            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # print('this is ',i,'layer x = ', residual.shape)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            #print('this is ',i,'layer TCN-a = ',filter.shape)
            filter = torch.tanh(filter)
            # print('this is ',i,'layer TCN-a_tanh = ', filter.shape)
            gate = self.gate_convs[i](residual)
            #print('this is ',i,'layer  TCN-b = ', gate.shape)
            gate = torch.sigmoid(gate)
            #print('this is ',i,'layer TCN-b_sigmoid = ', gate.shape)
            x = filter * gate
            # print('this is ',i,'layer filter + gate = ', x.shape)


            s = x
            s = self.skip_convs[i](s)
            # print('this is ',i,'layer s', s.shape)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0

            # print('this is ', i, 'layer skip1', skip.shape)
            skip = s + skip
            # print('this is ' ,i, 'layer skip2', skip.shape)


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            #  print('this is ' ,i, 'layer residual', x.shape)
            x = self.bn[i](x)   # 数据归一化

        x = F.relu(skip)
        # print('this is ', i, 'after relu', x.shape)
        x = F.relu(self.end_conv_1(x))
        # print('this is ', i, 'end_conv1', x.shape)
        x = self.end_conv_2(x)
        # print('this is ', i, 'layer output', x.shape)
        return x





