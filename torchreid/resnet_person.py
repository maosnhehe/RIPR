'''

Modified from huanghoujing's 
https://github.com/huanghoujing/person-reid-triplet-loss-baseline/blob/master/tri_loss/model/resnet.py
for building up flexible resnet50 architecture

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class selfAtten1(nn.Module):

  def __init__(self, cin, r=16):
    super(selfAtten1, self).__init__()
    self.conv_d = nn.Conv2d(cin, cin//r, kernel_size=1, stride=1)
    self.conv_u = nn.Conv2d(cin//r, cin, kernel_size=1, stride=1)
    
    self.bn_d = nn.BatchNorm2d(cin//r)
    self.bn_u = nn.BatchNorm2d(cin)
    
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.fill_(0)
        m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x):
    out = self.relu(self.bn_d(self.conv_d(x)))
    out = self.sigmoid(self.bn_u(self.conv_u(out)))
    return out


class selfAtten_se(nn.Module):

  def __init__(self, cin, r=16):
    super(selfAtten_se, self).__init__()
    self.conv1 = nn.Conv2d(cin, cin//r, kernel_size=(1, 1))
    self.conv2 = nn.Conv2d(cin//r, cin, kernel_size=(1, 1))
    self.bn1 = nn.BatchNorm2d(cin//r)

    self.bn2 = nn.BatchNorm2d(cin)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.fill_(0)
        m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x):
    n, c, h, w = x.shape

    out = F.avg_pool2d(x, (h, w))
    out = self.relu(self.bn1(self.conv1(out)))
    out = self.conv2(out)
    # out = self.bn2(out)
    out = self.sigmoid(out)
    return out

class selfAtten_cascade(nn.Module):
  def __init__(self, cin, r=16):
    super(selfAtten_cascade, self).__init__()
    self.ch = selfAtten_se(cin, r)
    self.sp = selfAtten1(cin, r)

  def forward(self, x):
    masksp = self.sp(x)
    out = x*masksp

    maskch = self.ch(out)
    out = maskch*out
    return out 



def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)



class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out



class ResNet(nn.Module):

  def __init__(self, block, layers, last_conv_stride=2, atten=selfAtten_cascade, r=16, fuse='avg'):

    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(
      block, 512, layers[3], stride=last_conv_stride)

    self.satten1 = atten(64*4, r)
    self.satten2 = atten(128*4, r)
    self.satten3 = atten(256*4, r)


    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
      nn.Conv2d(self.inplanes, planes * block.expansion,
            kernel_size=1, stride=stride, bias=False),
      nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)

    mask1s = self.satten1(x)

    x =  mask1s

    x = self.layer2(x)

    mask2s = self.satten2(x)

    x =  mask2s

    x = self.layer3(x)

    mask3s = self.satten3(x)

    x =  mask3s

    x = self.layer4(x)

    return x


def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  for key, value in list(state_dict.items()):
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict

def resnet50_buildup(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

  
  if pretrained:
    pretrained_state_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_state_dict = model.state_dict()
    load_my_state_dict(model_state_dict, pretrained_state_dict)
  

  return model

def load_my_state_dict(own_state_dict, pretrained_state_dict):
  for name, param in list(pretrained_state_dict.items()):
    if isinstance(param, Parameter):
      param = param.data
    own_state_dict[name].copy_(param)


class resnet50(nn.Module):
  def __init__(self, last_conv_stride=2, pretrained=True):
    super(resnet50, self).__init__()
    self.base = resnet50_buildup(pretrained=pretrained, last_conv_stride=last_conv_stride)
  def forward(self, input):
  # shape [N, C, H, W]
    output = self.base(input)
    flatten = F.avg_pool2d(output, output.size()[2:])
    # shape [N, C]
    output = flatten.view(flatten.size(0), -1)
    return output


class net(nn.Module):
    def __init__(self, n_id_class, last_conv_stride, fea_dim=2048):
        super(net, self).__init__()
        self.fea_dim = fea_dim
        self.net = resnet50(last_conv_stride)
        self.bn = nn.BatchNorm2d(fea_dim)
        self.fc = nn.Linear(fea_dim, n_id_class)

    def forward(self, x):
        embedding = self.get_embedding(x)
        out = self.fc(embedding)
        #return out, embedding
        return embedding

    def classify(self, x):
        return self.fc(x)
        
    def get_embedding(self, x):
        out = self.net(x)
        rev_flat = out.view(-1, self.fea_dim, 1, 1)
        out = self.bn(rev_flat)
        return out.view(-1, self.fea_dim)




class attentionlinear(nn.Module):
    def __init__(self,inputchannel,minchannel):                             
        super(attentionlinear, self).__init__()
        
        self.fc1 = nn.Linear(inputchannel, minchannel)
        self.fc2 = nn.Linear(minchannel,1)
        


    def forward(self, x) :
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)                 
        y=F.relu(self.fc1(x))
        y=F.sigmoid(self.fc2(y))
        y=y.view(y.size(0),y.size(1),1,1)
        return y




class DEC1(nn.Module):
    def __init__(self):                             
        super(DEC1, self).__init__()
        
        
        self.conv0 =nn.Conv2d(3,64,2,stride=2)
        self.conv1 = nn.Conv2d(64, 64, 5,padding=2)
        
        self.conv12 =nn.Conv2d(64,128,2,stride=2)
        self.conv2 = nn.Conv2d(128, 256, 3,padding=1)
        
        self.conv3 = nn.Conv2d(256,64,1)

        self.conv34 = nn.Conv2d(64,64, 3,padding=1)
        

        self.conv4 = nn.Conv2d(64, 256,1)
        # self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256,64,1)

        self.conv56 = nn.Conv2d(64,64, 3,padding=1)
        

        self.conv6 = nn.Conv2d(64, 256,1)
        # self.bn6 = nn.BatchNorm2d(256)

        self.dconv1 =nn.ConvTranspose2d(256,256,2,stride=2)
        self.conv9 = nn.Conv2d(256, 64, 3,padding=1)

        self.dconv2 =nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv10 = nn.Conv2d(64, 64, 3,padding=1)
        self.conv11 = nn.Conv2d(64, 3, 3,padding=1)
        


    def forward(self, x) :                  #origin:24*8*8

        o0=x
        x=self.conv0(x)
        x = F.relu(self.conv1(x))
        
        o1=x
        x=self.conv12(x)
        x = F.relu(self.conv2(x))
        o2=x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv34(x))
        x=self.conv4(x)
        # x = F.relu(self.bn4())
        x=x+o2
        o2=x
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv56(x))
        z=self.conv6(x)
        # x = F.relu(self.bn6(self.conv6(x)))
        x=z+o2
        
    
        x= F.relu(self.dconv1(x))
        x = F.relu(self.conv9(x))

        x=torch.cat((x,o1),1)
        x= F.relu(self.dconv2(x))
        x= F.relu(self.conv10(x))
        x = self.conv11(x)
        x=x+o0

        return x




class net2(nn.Module):
    def __init__(self, num_classes):
      super(net2, self).__init__()
      self.SRnet=DEC1()

      self.model1=net(n_id_class=num_classes, last_conv_stride=1)
      self.model2=net(n_id_class=num_classes, last_conv_stride=1)

      line_1=nn.ModuleList([attentionlinear(64,16) for i in range(2)])
      line_2=nn.ModuleList([attentionlinear(256,16) for i in range(2)])
      line_3=nn.ModuleList([attentionlinear(512,16) for i in range(2)])
      line_4=nn.ModuleList([attentionlinear(1024,16) for i in range(2)])
      line_5=nn.ModuleList([attentionlinear(2048,16) for i in range(2)])
      self.attmodel=nn.ModuleList([line_1,line_2,line_3,line_4,line_5])



      self.modelpart1=self.model1.net.base.conv1
      self.modelpart1=nn.Sequential(self.modelpart1,self.model1.net.base.bn1)
      self.modelpart1=nn.Sequential(self.modelpart1,self.model1.net.base.relu)
      self.modelpart1=nn.Sequential(self.modelpart1,self.model1.net.base.maxpool)
      self.modelpart2=self.model1.net.base.layer1
      self.modelpart3=self.model1.net.base.layer2
      self.modelpart4=self.model1.net.base.layer3
      self.modelpart5=self.model1.net.base.layer4

      self.model2part1=self.model2.net.base.conv1
      self.model2part1=nn.Sequential(self.model2part1,self.model2.net.base.bn1)
      self.model2part1=nn.Sequential(self.model2part1,self.model2.net.base.relu)
      self.model2part1=nn.Sequential(self.model2part1,self.model2.net.base.maxpool)
      self.model2part2=self.model2.net.base.layer1
      self.model2part3=self.model2.net.base.layer2
      self.model2part4=self.model2.net.base.layer3
      self.model2part5=self.model2.net.base.layer4

    def forward(self,x):
      y=self.SRnet(x)
      # z=self.model1(x)
      feat_1=self.modelpart1(y)
      feat_2=self.model2part1(y)
      at11=self.attmodel[0][0](feat_1)
      at21=self.attmodel[0][1](feat_2)
      feat=feat_1*at11+feat_2*at21


      feat_1=self.modelpart2(feat)
      feat_2=self.model2part2(feat)
      at12=self.attmodel[1][0](feat_1)
      at22=self.attmodel[1][1](feat_2)
      feat=feat_1*at12+feat_2*at22


      feat_1=self.modelpart3(feat)
      feat_2=self.model2part3(feat)
      at13=self.attmodel[2][0](feat_1)
      at23=self.attmodel[2][1](feat_2)
      feat=feat_1*at13+feat_2*at23


      feat_1=self.modelpart4(feat)
      feat_2=self.model2part4(feat)
      at14=self.attmodel[3][0](feat_1)
      at24=self.attmodel[3][1](feat_2)
      feat=feat_1*at14+feat_2*at24


      feat_1=self.modelpart5(feat)
      feat_2=self.model2part5(feat)
      at15=self.attmodel[4][0](feat_1)
      at25=self.attmodel[4][1](feat_2)
      zz=feat_1*at15+feat_2*at25
      # zz=x
      x = F.avg_pool2d(zz, zz.size()[2:])
      x = x.view(x.size(0), -1)
      rev_flat = x.view(-1, self.model1.fea_dim, 1, 1)
      out = self.model1.bn(rev_flat)
      out=out.view(-1, self.model1.fea_dim)
      # out=z
      if not self.training:
        return out
      # print(out.size())

      # print(self.model1.fc)
      prid = self.model1.fc(out)
      # print(y.size())
      return prid,out,y,at11,at12,at13,at14,at15,at21,at22,at23,at24,at25
