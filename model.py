

from typing import List
import torch.nn as nn
import logging
import warnings
import torch
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
from paraphrase.utils.data import FewShotDataset, FewShotSSLParaphraseDataset, FewShotSSLFileDataset
import numpy as np
import collections
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#点积计算相似度
def dot_similarity(x1, x2):
    return torch.matmul(x1, x2.t())


def squash(x):
    squared = torch.sum(x ** 2, dim=1).reshape(x.shape[0], -1)
    coeff = squared / (0.5 + squared) / torch.sqrt(squared + 1e-9)
    x = coeff * x
    return x




class Criterion(_Loss):
    def __init__(self, way=2, shot=5):
        super(Criterion, self).__init__()
        self.amount = way * shot

    def forward(self, probs,target):  # (Q,C) (Q)
        target_onehot = torch.zeros_like(probs)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        loss = torch.mean((probs/100.0 - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1)
        acc = torch.sum(target == pred).float() / target.shape[0]
        return acc, loss

#损失计算
class Contrastive_Loss(nn.Module):

    def __init__(self, tau=5.0):
        super(Contrastive_Loss, self).__init__()
        self.tau = tau

    def similarity(self, x1, x2):
        # # Gaussian Kernel
        # M = euclidean_dist(x1, x2)
        # s = torch.exp(-M/self.tau)

        # dot product
        M = dot_similarity(x1, x2)/self.tau
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_labels, *x):
        X = torch.cat(x, 0)
        len_ = batch_labels.size()[0]
        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        if len(x)==1:
            mask_i = torch.from_numpy(np.ones((len_, len_))).to(batch_labels.device)
        else:
            mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device) # sum over items in the numerator
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float()*mask_i # sum over items in the denominator
        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10)
        s_j = torch.clamp(s*mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        loss = torch.mean(log_p)

        return loss


class Induction(nn.Module):
    def __init__(self, n_classes, n_support, hidden_size, iterations):
        super(Induction, self).__init__()
        self.C = n_classes
        self.S = n_support
        self.H = hidden_size
        self.iterations = iterations
        self.W = torch.nn.Parameter(torch.randn(self.H, self.H))

    def forward(self, x):
        #初始化bij为0
        b_ij = torch.zeros(self.C, self.S).to(x)
        for _ in range(self.iterations):
            d_i = F.softmax(b_ij.unsqueeze(2), dim=1)  # (C,S,1)
            e_ij = torch.mm(x.reshape(-1, self.H), self.W).reshape(self.C, self.S, self.H)  # (C,S,H)
            c_i = torch.sum(d_i * e_ij, dim=1)  # (C,H)
            #c_i=c_i/||c_i||
            squared = torch.sum(c_i ** 2, dim=1).reshape(self.C, -1)
            coeff = 1 / torch.sqrt(squared + 1e-9)
            c_i = coeff * c_i

            distance = torch.bmm(e_ij, c_i.unsqueeze(2))  # (C,S,H)*(C,H,1)==(C,S,1)
            b_ij =  distance.squeeze(2)/100  #(C,S)
        c_i= torch.sum(d_i*x.reshape(self.C,self.S,self.H),dim=1)

        return c_i








#模型搭建
class ContrastNet(nn.Module):
    def __init__(self, config_name_or_path,iterations,n_classes,n_support,hidden_size=384,
                 metric="euclidean", max_len=64, super_tau=1.0):
        super(ContrastNet, self).__init__()
        #分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
        #encoder选择预训练模型
        self.encoder = AutoModel.from_pretrained(config_name_or_path,output_hidden_states=True).to(device)
        self.metric = metric
        self.max_len = max_len
        #计算相似度：欧几里得距离，余弦距离
        assert self.metric in ('euclidean', 'cosine')
        #交叉熵损失
        self.CE_loss=Criterion(n_classes,n_support)
        #对比监督对比损失
        self.contrast_loss = Contrastive_Loss(super_tau)
        self.induction = Induction(n_classes, n_support+1, 2 * hidden_size, iterations).to(device)
        self.warmed: bool = False

    #encode负责将句子变为句向量，输入为句子的列表
    def encode(self, sentences: List[str]):
        if self.warmed:
            padding = True
        else:
            padding = "max_length"
            self.warmed = True
        #分词模块，将文本转为编码
        #text-》tensor
        batch = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            padding=padding
        )
        #拿到key和value.to(device)将value放入gpu
        batch = {k: v.to(device) for k, v in batch.items()}



        #直接使用CLS的特征表示句子特征
        hidden = self.encoder.forward(**batch).last_hidden_state
        return hidden[:,0,:]


        # #使用最后两层的平均池化表示句子特征
        #
        # hidden_states=self.encoder.forward(**batch).hidden_states
        # attention_mask=batch["attention_mask"]
        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states[-1].size()).float()
        # sum_mask = input_mask_expanded.sum(1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        #
        # token_embeddings_last1 = hidden_states[-1]
        # sum_embeddings_last1 = torch.sum(token_embeddings_last1 * input_mask_expanded, 1)
        # sum_embeddings_last1 = sum_embeddings_last1 / sum_mask
        #
        # token_embeddings_last2 = hidden_states[-2]
        # sum_embeddings_last2 = torch.sum(token_embeddings_last2 * input_mask_expanded, 1)
        # sum_embeddings_last2 = sum_embeddings_last2 / sum_mask
        # return (sum_embeddings_last1 + sum_embeddings_last2) / 2



    #计算query到各个proto的距离，从而找出最近的类
    def pred_proto(self, query, proto):
        s = dot_similarity(query, proto)
        _, y_pred = s.max(1)
        return y_pred




    def loss(self, sample, supervised_loss_share: float = 0):

        xs = sample['xs']  # support
        xq = sample['xq']  # query
        #xs和xq为抽取的五个label的列表
        n_class = len(xs)
        assert len(xq) == n_class
        #n_support为抽取的某一label的样本数量
        n_support = len(xs[0])
        n_query = len(xq[0])

        #支撑集y标签
        support_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support, 1).long()
        support_inds = Variable(support_inds, requires_grad=False).to(device)

        query_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        query_inds = Variable(query_inds, requires_grad=False).to(device)


        #将S和Q中的文本拿出
        supports = [item["sentence"] for xs_ in xs for item in xs_]
        queries = [item["sentence"] for xq_ in xq for item in xq_]
        #将S和Q拼接[x1,x2....xs,xs+1,xs+2...,xs+q]
        x = supports + queries

        #z为编码后的隐层状态
        z = self.encode(x)
        #z的特征维数
        z_dim = z.size(-1)

        #获取S和Q的句子表示
        z_support = z[:len(supports)]
        z_query = z[len(supports):len(supports) + len(queries)]

        # #固定原型
        # z_support_proto = z_support.view(n_class, n_support, z_dim)[:,-1,:]
        #计算supoort集和固定类描述的平均作为原型
        # z_support_proto = z_support.view(n_class, n_support, z_dim).mean(dim=[1])



        # supervised contrastive loss
        z_query_in = z_query
        z_support_in = z_support
        contrast_labels = torch.cat([support_inds.reshape(-1),query_inds.reshape(-1)],dim=0)
        #对比监督损失计算
        supervised_loss = self.contrast_loss(contrast_labels, z_query_in, z_support_in)


        #根据support集归纳出类别
        z_class=self.induction(z_support)#(5,768)
        #query与class的向量做点积得到相似度分数
        indution_probs=dot_similarity(z_query.reshape(n_class*n_query,z_dim),z_class)
        #最大值为预测的类别
        _,indution_pred= indution_probs.max(1)
        #计算正确率
        indution_acc=torch.eq(indution_pred, query_inds.reshape(-1)).float().mean()
        #计算CrossEntropyLoss
        loss_item = torch.nn.CrossEntropyLoss()
        ce_loss = loss_item(indution_probs, query_inds.reshape(-1))


        final_loss = supervised_loss_share*supervised_loss+(1-supervised_loss_share)*ce_loss




        return final_loss, {
            "metrics": {
                "acc_induction": indution_acc.item(),
                "CE_loss":ce_loss.item(),
                "supervised_loss": supervised_loss.item(),
                "total_loss": final_loss.item(),
            },
            "target": query_inds
        }
    def train_step(self, optimizer, episode, supervised_loss_share: float):
        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss, loss_dict = self.loss(episode, supervised_loss_share=supervised_loss_share)
        loss.backward()
        optimizer.step()
        return loss, loss_dict


    def test_step(self, dataset: FewShotDataset, n_episodes: int = 1000):
        metrics = collections.defaultdict(list)

        self.eval()
        for i in range(n_episodes):
            episode,_ = dataset.get_episode()

            with torch.no_grad():
                loss, loss_dict = self.loss(episode, supervised_loss_share=1)

            for k, v in loss_dict["metrics"].items():
                metrics[k].append(v)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }

    def tsne(self,dataset:FewShotDataset,n_episodes: int = 100):
        self.eval()
        for i in range(n_episodes):
            episode, _ = dataset.get_episode()
            xs = episode['xs']  # support
            xq = episode['xq']  # query
            # xs和xq为抽取的五个label的列表
            n_class = len(xs)
            assert len(xq) == n_class


            # 将S和Q中的文本拿出
            supports = [item["sentence"] for xs_ in xs for item in xs_]
            y_supports= [item["label"] for xs_ in xs for item in xs_]
            queries = [item["sentence"] for xq_ in xq for item in xq_]
            y_queries = [item["label"] for xq_ in xq for item in xq_]
            # 将S和Q拼接[x1,x2....xs,xs+1,xs+2...,xs+q]
            x = supports + queries
            y= y_supports + y_queries
            # z为编码后的隐层状态
            z = self.encode(x)


            # 获取S和Q的句子表示
            z_support = z[:len(supports)]
            z_query = z[len(supports):len(supports) + len(queries)]

            # #固定原型
            # z_support_proto = z_support.view(n_class, n_support, z_dim)[:,-1,:]
            # 计算supoort集和固定类描述的平均作为原型
            # z_support_proto = z_support.view(n_class, n_support, z_dim).mean(dim=[1])

            # 根据support集归纳出类别
            z_class = self.induction(z_support)  # (5,768)










        return 0








