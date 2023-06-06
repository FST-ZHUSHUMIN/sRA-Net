import torch
from torch import nn

from torch.nn import functional as F
from torch.nn.modules import activation
from src import const
from backbone.base_networks import VGG16Extractor
import numpy as np
import math



class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=1, gamma_pos=0, clip=0, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss
    
# def BCE_loss(pos_weight, output, ground, mask):
#     ones = torch.ones(ground.size()).to(const.device)
#     logx = -(pos_weight * ground * torch.log(output) + (ones-ground)* output* torch.log(ones-output)) * (mask)
#     #print(logx.size()) 
#     loss = torch.mean(torch.sum(logx, axis = 1))
#     return loss

class df_sRAnet(nn.Module):
    def __init__(self, num_attr=1000, num_labels=48,use_lmt=True,layers=3,heads=4,dropout=0.1,int_loss=0,no_x_features=False):
        super(df_sRAnet, self).__init__()
        self.use_lmt = use_lmt
 
        # ResNet backbone
        self.backbone = VGG16Extractor()
        hidden = 512 # this should match the backbone output feature size
        num_add_token = num_attr + 2 # 多加的一个作为label 分类， 剩下的每一个 token 是一个attribute
        self.num_add_token = num_add_token
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_add_token)).view(1,-1).long()
        self.label_lt = nn.Embedding(num_add_token, hidden, padding_idx=None)
        
        # State Embeddings
        self.known_label_lt = nn.Embedding(2, hidden, padding_idx=0)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.attr_linear = nn.Linear(hidden,num_attr)
        self.label_linear = nn.Linear(hidden, num_labels)
        self.type_linear = nn.Linear(hidden, 3)


        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
      
        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.attr_linear.apply(weights_init)
        self.label_linear.apply(weights_init)
        self.type_linear.apply(weights_init)
        # self.output_linear.apply(weights_init)

    def forward(self, sample):
        images = sample['image']
        const_label_input = self.label_input.repeat(images.size(0),1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)
        #print('data0',self.known_label_lt.weight.data[0])
        #print('data1',self.known_label_lt.weight.data[1])
        # print('images input', images.size())
        features = self.backbone(images)
        features = features['conv5_3']

        features = features.view(features.size(0),features.size(1),-1).permute(0,2,1) 
        #print(mask.size())
        
        # Concat image and label embeddings
        embeddings = init_label_embeddings

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        attns = []

        for layer in self.self_attn_layers:
            embeddings,attn = layer(features, embeddings,mask=None)
            attns += attn.detach().unsqueeze(0)

        # Readout each label embedding using a linear layer
        attr_embeddings = embeddings[:, 0:-2,:]
        label_embeddings = embeddings[:, -2, :]
        type_embeddings = embeddings[:, -1, :]
        
        #print('size of attr embeddings: ', attr_embeddings.size())
        attr_feature = self.attr_linear(attr_embeddings)
        #print('size of attr features: ', attr_feature.size())

        label_feature = self.label_linear(label_embeddings) 
        
        type_feature = self.type_linear(type_embeddings)
        
        # print('size of label feature: ', label_feature.size())
        diag_mask = torch.eye(attr_feature.size(1)).unsqueeze(0).repeat(attr_feature.size(0),1,1).cuda()
        multiply = attr_feature*diag_mask
        #print('attr_feature: ', attr_feature.size(), 'diag_mask: ', diag_mask.size(), 'multiplay: ', multiply.size())
        
        output = {} 
        attr_output = (attr_feature*diag_mask).sum(-1)
        output['attr_output'] = attr_output
        output['category_output'] = label_feature.squeeze(1)
        output['type_output'] = type_feature.squeeze(1)
        # print('output[category_output]: ', output['category_output'].size())
        # exit()
        return output
        
        
    def cal_loss(self, sample, output):

        self.pos_weight = torch.ones(output['attr_output'].size())*8
        #self.sigmoid = nn.Sigmoid()
        self.label_loss_func = nn.CrossEntropyLoss()
        self.type_loss_func =nn.CrossEntropyLoss()
        self.attr_loss = AsymmetricLossOptimized()
        #attr_log = self.sigmoid(output['attr_output'].float())
        #attr_loss = BCE_loss(self.pos_weight.to(const.device), attr_log, sample['attr'].float(), sample['mask'])
        
        attr_loss = self.attr_loss(output['attr_output'], sample['attr'].long())
        category_loss = self.label_loss_func(output['category_output'], sample['category_label'].long())
        type_loss = self.type_loss_func(output['type_output'], sample['category_type'])
        
        all_loss = 2 * attr_loss + category_loss #+ type_loss
        
        # print(attr_loss, category_loss, type_loss)
        loss = {
            'all': all_loss,
            'category_loss':category_loss ,
            'attr_loss': attr_loss.item(),
            'weighted_attr_loss':   2*attr_loss.item(),
            'weighted_category_loss': category_loss.item()
        }
        return loss


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model*2, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(dim_feedforward, d_model)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, attributes, src_mask= None, src_key_padding_mask = None):
        # print('initional size of attributes: ', attributes.size())    #1001,16,512 
        src1,attn = self.self_attn(attributes, attributes, attributes, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        src = attributes + self.dropout1(src1) 
        src = self.norm1(src)

        src2, attn = self.cross_attn(src, img_features, img_features, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)
        
        ca = torch.cat([attributes, src2], dim = 2)
        alphac = self.sigmoid(self.linear2(self.dropout(self.activation(self.linear1(ca)))))
        src3 = torch.mul(src2, alphac)
        src = attributes + src3
        src = self.norm2(src)     
        #src = src2 
        # src = self.norm2(src2)
        
        src4 = self.linear4(self.dropout(self.activation(self.linear3(src))))
        src = src + self.dropout3(src4)
        src = self.norm3(src)
        return src, attn


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead = 4,dropout=0.1):
        super().__init__()
        # self.transformer_layer = TransformerEncoderLayer(d_model, nhead, d_model*1, dropout=dropout, activation='relu')
        # self.transformer_layer = TransformerEncoderLayer(d_model, nhead, d_model, dropout=dropout, activation='gelu') 
        self.transformer_layer = TransformerDecoderLayer(d_model, nhead, d_model, dropout=dropout, activation = 'gelu')

    def forward(self, img_features, attributes, mask=None):
        attn = None
        attributes =  attributes.transpose(0,1)  
        img_features = img_features.transpose(0,1)
        x,attn = self.transformer_layer(img_features, attributes,src_mask=mask)
        x=x.transpose(0,1)
        return x,attn

def weights_init(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def pos_init(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        print(module.weight.size())
        module.weight.data[0].fill_(1)
        module.weight.data[1].fill_(1)
        #module.weight.data[0].detach = True
        
        module.weight.data[1].requires_grad = True
        print(module.weight.data[0].requires_grad)
        print(module.weight.data[1].requires_grad)
    #print(module.weight.size())
    #module.weight.data.fill_(0.01)
    # stdv = 1. / math.sqrt(module.weight.size(1))
    # module.weight.data.uniform_(-stdv, stdv)
    
# def custom_replace(tensor,on_neg_1,on_zero, on_one):
#     res = tensor.clone()
#     res[tensor == -1] = on_neg_1
#     res[tensor == 0] = on_zero
#     res[tensor == 1] = on_one
#     return res

def custom_replace(tensor,on_zero, on_one):
    res = tensor.clone()
    res[tensor == 0] = on_zero
    res[tensor == 1] = on_one
    return res

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
