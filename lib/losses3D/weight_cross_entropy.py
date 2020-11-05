from lib.losses3D.basic import *
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        # import ipdb;ipdb.set_trace()
        N,_,h,w,d = target.shape
        target_new = torch.zeros(N,h,w,d).cuda()
        N_ind, C_ind, h_ind, w_ind, d_ind = torch.where(target!=0)
        target_new[(N_ind, h_ind, w_ind, d_ind)] = C_ind.float()
        target = target_new.long()
        return torch.nn.functional.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        # import ipdb;ipdb.set_trace()
        input = torch.nn.functional.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = torch.autograd.Variable(nominator / denominator, requires_grad=False)
        return class_weights
