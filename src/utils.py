import torch
import pandas as pd
import numpy as np
from src import const
import importlib
import argparse
import math

class df_Evaluator(object):

    def __init__(self, category_topk=(1, 3, 5), attr_topk=(1, 3, 5), mode = 'attribute'):  #['attribute','addcategory','addlandmark']
        self.mode = mode
        self.category_topk = category_topk
        self.attr_topk = attr_topk
        self.reset()
        # with open(const.base_df_path + 'Anno/list_attr_cloth.txt') as f:
        #     ret = []
        #     f.readline()
        #     f.readline()
        #     for line in f:
        #         line = line.split(' ')
        #         while line[-1].strip().isdigit() is False:
        #             line = line[:-1]
        #         ret.append([
        #             ' '.join(line[0:-1]).strip(),
        #             int(line[-1])
        #         ])
        # attr_type = pd.DataFrame(ret, columns=['attr_name', 'type'])
        # attr_type['attr_index'] = ['attr_' + str(i) for i in range(1000)]
        # attr_type.set_index('attr_index', inplace=True)
        # self.attr_type = attr_type
        attr_type = pd.read_csv(const.base_df_path + 'Anno/abstract_attr_cloth.csv')
        attr_type['attr_index'] = ['attr_' + str(i) for i in range(1000)]
        attr_type.set_index('attr_index', inplace = True)
        self.attr_type = attr_type

    def reset(self):

        if self.mode == 'attribute':
            self.attr_group_gt = np.array([0.] * 5)
            self.attr_group_tp = np.zeros((5, len(self.attr_topk)))
            self.attr_all_gt = 0
            self.attr_all_tp = np.zeros((len(self.attr_topk),))

        if self.mode == 'addcategory':
            self.category_accuracy = []
            self.attr_group_gt = np.array([0.] * 5)
            self.attr_group_tp = np.zeros((5, len(self.attr_topk)))
            self.attr_all_gt = 0
            self.attr_all_tp = np.zeros((len(self.attr_topk),))

        if self.mode == 'addlandmark':
            self.category_accuracy = []
            self.lm_vis_count_all = np.array([0.] * 8)
            self.lm_dist_all = np.array([0.] * 8)

            self.attr_group_gt = np.array([0.] * 5)
            self.attr_group_tp = np.zeros((5, len(self.attr_topk)))
            self.attr_all_gt = 0
            self.attr_all_tp = np.zeros((len(self.attr_topk),))

    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in self.category_topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100 / batch_size))
            for i in range(len(res)):
                res[i] = res[i].cpu().numpy()[0] / 100
            self.category_accuracy.append(res)


    def attr_count(self, output, sample):
        attr_group_gt = np.array([0.] * 5)
        attr_group_tp = np.zeros((5, len(self.attr_topk)))
        attr_all_tp = np.zeros((len(self.attr_topk),))

        target = sample['attr'].cpu().numpy()
        target = np.split(target, target.shape[0])
        target = [x[0, :] for x in target]

        pred = output['attr_output'].cpu().detach().numpy()
        pred = np.split(pred, pred.shape[0])
        pred = [x[0, :] for x in pred]

        for batch_idx in range(len(target)):
            result_df = pd.DataFrame([target[batch_idx], pred[batch_idx]],
                                     index=['target', 'pred'], columns=['attr_' + str(i) for i in range(1000)])
            result_df = result_df.transpose()
            result_df = result_df.join(self.attr_type[['type']])
            ret = []
            for i in range(1, 6):
                ret.append(result_df[result_df['type'] == i]['target'].sum())
            # 计算得到了对于一张图片，每一个group的 target 的和
            attr_group_gt += np.array(ret)   # 存储和
            
            ret = []
            result_df = result_df.sort_values('pred', ascending=False)  #总排序

            # print(result_df)
            attr_all_tp += np.array([
                result_df.head(k)['target'].sum() for k in self.attr_topk
            ])

            for i in range(1, 6):
                sort_df = result_df[result_df['type'] == i]
                ret.append([
                    sort_df.head(k)['target'].sum() for k in self.attr_topk
                ])
            attr_group_tp += np.array(ret)

        self.attr_group_gt += attr_group_gt
        self.attr_group_tp += attr_group_tp
        self.attr_all_gt += attr_group_gt.sum()
        self.attr_all_tp += attr_all_tp


    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        landmark_dist = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * output['lm_pos_output'] - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def add(self, output, sample):
        if self.mode == 'attribute':
            self.attr_count(output, sample)
        
        if self.mode == 'addcategory':
            self.attr_count(output, sample)
            self.category_topk_accuracy(output['category_output'], sample['category_label'])
        
        if self.mode == 'addlandmark':
            self.attr_count(output, sample)
            self.landmark_count(output, sample)
            self.category_topk_accuracy(output['category_output'], sample['category_label'])

    def evaluate(self):
        attr_group_recall = {}
        attr_recall_real = {}
        attr_recall_fake = {}
        ret = {}

        for i, top_n in enumerate(self.attr_topk):
            attr_group_recall[top_n] = self.attr_group_tp[..., i] / self.attr_group_gt
            attr_recall_real[top_n] = self.attr_all_tp[i] / self.attr_all_gt
            attr_recall_fake[top_n] = self.attr_group_tp[...,i].sum() / self.attr_all_gt
        
        ret['attr_group_recall'] = attr_group_recall
        ret['attr_recall_real'] = attr_recall_real
        ret['attr_recall_fake'] = attr_recall_fake

        if self.mode == 'addcategory':
            category_accuracy = np.array(self.category_accuracy).mean(axis=0)
            category_accuracy_topk = {}
            for i, top_n in enumerate(self.category_topk):
                category_accuracy_topk[top_n] = category_accuracy[i]
            ret['category_accuracy_topk'] = category_accuracy_topk
        
        if self.mode == 'addlandmark':
            category_accuracy = np.array(self.category_accuracy).mean(axis=0)
            category_accuracy_topk = {}
            for i, top_n in enumerate(self.category_topk):
                category_accuracy_topk[top_n] = category_accuracy[i]
            ret['category_accuracy_topk'] = category_accuracy_topk

            lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
            lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()
            ret['lm_individual_dist'] = lm_individual_dist
            ret['lm_dist'] = lm_dist

        return ret

class test_evaluate_attr(object):
    
    def __init__(self, difficult_examples=False):
        super(test_evaluate_attr, self).__init__()
        self.reset()                                     #init scores and targets
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.attr_scores = torch.FloatTensor(torch.FloatStorage())
        self.attr_targets = torch.LongTensor(torch.LongStorage())
        self.cate_scores = torch.LongTensor(torch.LongStorage())

    def add(self, attr_output, attr_target, cate_output, cate_target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(attr_output):
            output = torch.from_numpy(attr_output)
        if not torch.is_tensor(attr_target):
            target = torch.from_numpy(attr_target)
        
        if not torch.is_tensor(cate_output):
                output = torch.from_numpy(cate_output)
        if not torch.is_tensor(cate_target):
            target = torch.from_numpy(cate_target)
        
 
        
        if attr_output.dim() == 1:
            attr_output = attr_output.view(-1, 1)
        else:
            assert attr_output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if attr_target.dim() == 1:
            attr_target = attr_target.view(-1, 1)
        else:
            assert attr_target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        
        if cate_output.dim() == 1:
                cate_output = output.view(-1, 1)
        else:
            assert cate_output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
       # if target.dim() == 1:
        #    target = target.view(-1, 1)
        #else:
         #   assert target.dim() == 2, \
          #      'wrong target size (should be 1D or 2D with one column \
           #     per class)'
        
        
        if self.attr_scores.numel() > 0:
            assert attr_target.size(1) == self.attr_targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.attr_scores.storage().size() < self.attr_scores.numel() + attr_output.numel():
            new_size = math.ceil(self.attr_scores.storage().size() * 1.5)
            self.attr_scores.storage().resize_(int(new_size + attr_output.numel()))
            self.attr_targets.storage().resize_(int(new_size + attr_output.numel()))
            self.cate_scores.storage().resize_(int(new_size + attr_output.numel()))

        # store scores and targets
        offset = self.attr_scores.size(0) if self.attr_scores.dim() > 0 else 0
        self.attr_scores.resize_(offset + attr_output.size(0), attr_output.size(1))
        self.cate_scores.resize_(offset + cate_output.size(0), cate_output.size(1))
        
        self.attr_targets.resize_(offset + attr_target.size(0), attr_target.size(1))
        
        self.attr_scores.narrow(0, offset, attr_output.size(0)).copy_(attr_output)
        self.cate_scores.narrow(0, offset, cate_output.size(0)).copy_(cate_output)
        self.attr_targets.narrow(0, offset, attr_target.size(0)).copy_(attr_target)

    def overall(self):
        if self.attr_scores.numel() == 0:
            return 0
        attr_scores = self.attr_scores.cpu().numpy()
        attr_targets = self.attr_targets.cpu().numpy()
        #cate_scores = self.cate_scores.cpu().numpy()
        cate_scores = self.cate_scores.cpu()
                
        attr_targets[attr_targets == -1] = 0
        
        return self.evaluation(attr_scores, cate_scores)


    def evaluation(self, scores_, cate_):
        n, n_class = scores_.shape
        Nc, P, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        scores1 = scores_[:, 670]
        #print(scores1)
        scores1[scores1 >=0 ] = 1
        scores1[scores1 <0] = 0
        print(scores1)
        print(cate_.shape)
        _,ind = cate_.topk(1,1,True,True)
        ind = ind.t().squeeze().numpy()
        #ind = np.array(np.max(cate_, axis = 1))
        #print(ind)
        ind[ind != 2] = 0
        ind[ind == 2] = 1
        print(ind)
        count = np.sum(scores1*ind)
        p = count/n
        print(p,count,n)
        
        exit()
        scores2 = scores_[:, 671]
        scores =  np.sum((scores1>=0) * (scores2 >= 0))
        p = scores/n
        print(p,scores,n)
        #print(scores1, scores2)
        
        return p

class test_evaluate_sweater(object):
    
    def __init__(self, difficult_examples=False):
        super(test_evaluate_sweater, self).__init__()
        self.reset()                                     #init scores and targets
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
    
    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, P, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        scores1 = scores_[:, 670]
        scores2 = scores_[:, 612]
        scores =  np.sum((scores1>=0) * (scores2 >= 0))
        p = scores/n
        print(p,scores,n)
        #print(scores1, scores2)
        return p

class iM_Evaluator_group(object):

    def __init__(self, category_topk=(1, 3, 5), attr_topk=(1,3,5), mode = 'attribute'):  #['attribute','addcategory','addlandmark']
        self.mode = mode
        self.category_topk = category_topk
        self.attr_topk = attr_topk
        self.reset()
        attr_type = pd.read_csv(const.base_im_path + 'anno_group.csv')
        attr_type['attr_index'] = ['attr_' + str(i) for i in range(228)]
        attr_type.set_index('attr_index', inplace = True)
        self.attr_type = attr_type

    def reset(self):

        if self.mode == 'attribute':
            self.attr_group_gt = np.array([0.] * 8)
            self.attr_group_tp = np.zeros((8, len(self.attr_topk)))
            self.attr_all_gt = 0
            self.attr_all_tp = np.zeros((len(self.attr_topk),))

        if self.mode == 'addcategory':
            self.category_accuracy = []
            self.attr_group_gt = np.array([0.] * 7)
            self.attr_group_tp = np.zeros((7, len(self.attr_topk)))
            self.attr_all_gt = 0
            self.attr_all_tp = np.zeros((len(self.attr_topk),))

        if self.mode == 'addlandmark':
            self.category_accuracy = []
            self.lm_vis_count_all = np.array([0.] * 7)
            self.lm_dist_all = np.array([0.] * 7)

            self.attr_group_gt = np.array([0.] * 7)
            self.attr_group_tp = np.zeros((7, len(self.attr_topk)))
            self.attr_all_gt = 0
            self.attr_all_tp = np.zeros((len(self.attr_topk),))
    
    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in self.category_topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100 / batch_size))
            for i in range(len(res)):
                res[i] = res[i].cpu().numpy()[0] / 100
            self.category_accuracy.append(res)


    def attr_count(self, output, sample):
        attr_group_gt = np.array([0.] * 8)
        attr_group_tp = np.zeros((8, len(self.attr_topk)))
        attr_all_tp = np.zeros((len(self.attr_topk),))

        target = sample['attr'].cpu().numpy()
        target = np.split(target, target.shape[0])
        target = [x[0, :] for x in target]

        pred = output['attr_output'].cpu().detach().numpy()
        pred = np.split(pred, pred.shape[0])
        pred = [x[0, :] for x in pred]

        for batch_idx in range(len(target)):
            result_df = pd.DataFrame([target[batch_idx], pred[batch_idx]],
                                     index=['target', 'pred'], columns=['attr_' + str(i) for i in range(228)])

            result_df = result_df.transpose()
            result_df = result_df.join(self.attr_type[['taskId']])
            ret = []
            for i in range(1, 9):
                ret.append(result_df[result_df['taskId'] == i]['target'].sum())
            # 计算得到了对于一张图片，每一个group的 target 的和
            attr_group_gt += np.array(ret)   # 存储和
            
            ret = []
            result_df = result_df.sort_values('pred', ascending=False)  #总排序

            # print(result_df)
            attr_all_tp += np.array([
                result_df.head(k)['target'].sum() for k in self.attr_topk
            ])

            for i in range(1, 9):
                sort_df = result_df[result_df['taskId'] == i]
                ret.append([
                    sort_df.head(k)['target'].sum() for k in self.attr_topk
                ])
            attr_group_tp += np.array(ret)

        self.attr_group_gt += attr_group_gt
        self.attr_group_tp += attr_group_tp
        self.attr_all_gt += attr_group_gt.sum()
        self.attr_all_tp += attr_all_tp


    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        landmark_dist = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * output['lm_pos_output'] - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def add(self, output, sample):
        if self.mode == 'attribute':
            self.attr_count(output, sample)
        
        if self.mode == 'addcategory':
            self.attr_count(output, sample)
            self.category_topk_accuracy(output['category_output'], sample['category_label'])
        
        if self.mode == 'addlandmark':
            self.attr_count(output, sample)
            self.landmark_count(output, sample)
            self.category_topk_accuracy(output['category_output'], sample['category_label'])

    def evaluate(self):
        attr_group_recall = {}
        attr_recall_real = {}
        # attr_recall_fake = {}
        ret = {}

        for i, top_n in enumerate(self.attr_topk):
            attr_group_recall[top_n] = self.attr_group_tp[..., i] / self.attr_group_gt
            attr_recall_real[top_n] = self.attr_all_tp[i] / self.attr_all_gt
            # attr_recall_fake[top_n] = self.attr_group_tp[...,i].sum() / self.attr_all_gt
        
        ret['attr_group_recall'] = attr_group_recall
        ret['attr_recall_real'] = attr_recall_real
        # ret['attr_recall_fake'] = attr_recall_fake

        if self.mode == 'addcategory':
            category_accuracy = np.array(self.category_accuracy).mean(axis=0)
            category_accuracy_topk = {}
            for i, top_n in enumerate(self.category_topk):
                category_accuracy_topk[top_n] = category_accuracy[i]
            ret['category_accuracy_topk'] = category_accuracy_topk
        
        if self.mode == 'addlandmark':
            category_accuracy = np.array(self.category_accuracy).mean(axis=0)
            category_accuracy_topk = {}
            for i, top_n in enumerate(self.category_topk):
                category_accuracy_topk[top_n] = category_accuracy[i]
            ret['category_accuracy_topk'] = category_accuracy_topk

            lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
            lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()
            ret['lm_individual_dist'] = lm_individual_dist
            ret['lm_dist'] = lm_dist

        return ret


class iMaterialist_Evaluator(object):
    def __init__(self, difficult_examples=False):
        super(iMaterialist_Evaluator, self).__init__()
        self.reset()                                     #init scores and targets
        self.difficult_examples = difficult_examples
        # attr_type = pd.read_csv('/home/wangsibo_daniel/zhushumin-project/DeepFashionmaster/dataset/imaterialist-challenge-fashion-2018/anno_group.csv')
        # attr_type['attr_index'] = ['attr_' + str(i) for i in range(228)]
        # attr_type.set_index('attr_index', inplace = True)
        # self.attr_type = attr_type

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]   # 针对第K个属性的所有images scores
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = iMaterialist_Evaluator.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):  #mAP 是各类别AP的平均值  #AP 已经检查过，确定没有问题
        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.

        for i in indices:
            label = target[i]

            if label == 1:
                pos_count += 1
            
            total_count += 1
            
            if label == 1:
                precision_at_i += pos_count / total_count

        if int(pos_count) == 0:
            precision_at_i /= 1
            
        else:
            precision_at_i /= pos_count

        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        scores[scores >= 0] = 1
        scores[scores < 0] = -1

        return self.evaluation(scores, targets)

    def overall_topk(self, k):  # topK已经检查过，确定没有问题
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()

        tmp = self.scores.cpu().numpy()

        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
                
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)

        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            # print(scores == 1)
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)  #计算实际上在类别k上分数大于0的样本个数
            Nc[k] = np.sum(targets * (scores >= 0))
        
        Np[Np == 0] = 1
        # print('Np: ',np.sum(Np),'Ng: ',np.sum(Ng), 'Nc: ', np.sum(Nc))
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)
        # CP = np.sum(Nc / Np) / n_class
        # CR = np.sum(Nc / Ng) / n_class
        # CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1


class DARN_Evaluator(object):
    def __init__(self, difficult_examples=False):
        super(DARN_Evaluator, self).__init__()
        self.reset()                                     #init scores and targets
        self.difficult_examples = difficult_examples
        attr_type = pd.read_csv('/home/wangsibo_daniel/zhushumin-project/DeepFashionmaster/dataset/DARN/list_attr_cloth.csv')
        attr_type['attr_index'] = ['attr_' + str(i) for i in range(178)]
        attr_type.set_index('attr_index', inplace = True)
        self.attr_type = attr_type

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]   # 针对第K个属性的所有images scores
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = iMaterialist_Evaluator.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):  #mAP 是各类别AP的平均值  #AP 已经检查过，确定没有问题
        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.

        for i in indices:
            label = target[i]

            if label == 1:
                pos_count += 1
            
            total_count += 1
            
            if label == 1:
                precision_at_i += pos_count / total_count

        if int(pos_count) == 0:
            precision_at_i /= 1
            
        else:
            precision_at_i /= pos_count

        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        scores[scores >= 0] = 1
        scores[scores < 0] = -1

        return self.evaluation(scores, targets)

    def overall_topk(self, k):  # topK已经检查过，确定没有问题
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()

        tmp = self.scores.cpu().numpy()

        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
                
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)

        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            # print(scores == 1)
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)  #计算实际上在类别k上分数大于0的样本个数
            Nc[k] = np.sum(targets * (scores >= 0))
        
        # Np[Np == 0] = 1
        # print('Np: ',np.sum(Np),'Ng: ',np.sum(Ng), 'Nc: ', np.sum(Nc))
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        # CP = np.sum(Nc / Np) / n_class
        # CR = np.sum(Nc / Ng) / n_class
        # CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1


class LandmarkEvaluator(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()
        landmark_dist = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * output['lm_pos_output'].cpu().numpy() - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def add(self, output, sample):
        self.landmark_count(output, sample)

    def evaluate(self):
        lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()
        return {
            'category_accuracy_topk': {},
            'attr_group_recall': {},
            'attr_recall': {},
            'lm_individual_dist': lm_individual_dist,
            'lm_dist': lm_dist,
        }


def merge_const(module_name):
    new_conf = importlib.import_module(module_name)
    for key, value in new_conf.__dict__.items():
        if not(key.startswith('_')):
            setattr(const, key, value)
            print('override', key, value)


def parse_args_and_merge_const():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='', type=str)
    args = parser.parse_args()
    if args.conf != '':
        merge_const(args.conf)


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()                                     #init scores and targets
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]   # 针对第K个属性的所有images scores
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=False):  #mAP 是各类别AP的平均值

        # sort examples

        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]

            if label == 1:
                pos_count += 1

            total_count += 1
            
            if label == 1:
                precision_at_i += pos_count / total_count

        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
                
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)  #计算实际上在类别k上分数大于0的样本个数
            Nc[k] = np.sum(targets * (scores >= 0))
        
        # Np[Np == 0] = 1
        print('Np: ',np.sum(Np),'Ng: ',np.sum(Ng), 'Nc: ', np.sum(Nc))
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1
