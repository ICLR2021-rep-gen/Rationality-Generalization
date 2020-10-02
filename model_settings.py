import torch
import torch.nn as nn
from model import Model

class Net(nn.Module):
    def __init__(self, num_class, args, feature_size=None):
        super(Net, self).__init__()

        self.from_features = args.from_features

        if not self.from_features:
            self.f = Model(settings=args).f
            self.h_width = Model(settings=args).h_width
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        else:
            assert feature_size is not None
            self.h_width = feature_size

        ## Evaluation network
        if args.eval_arch is None:
            if args.eval_type=='linear':
                self.eval_net = nn.Linear(self.h_width, num_class, bias=True)
            elif args.eval_type=='fc':
                if not args.eval_nl_bn:
                    self.eval_net = nn.Sequential(nn.Linear(self.h_width, args.eval_nl_width, bias=True), 
                                   nn.ReLU(inplace=True), nn.Linear(args.eval_nl_width, num_class, bias=True))
                else:
                    self.eval_net = nn.Sequential(nn.Linear(self.h_width, args.eval_nl_width, bias=False), nn.BatchNorm1d(args.eval_nl_width), nn.ReLU(inplace=True), nn.Linear(args.eval_nl_width, num_class, bias=True))

    def forward(self, x):
        if not self.from_features:
            x = self.f(x)
            feature = torch.flatten(x, start_dim=1)
        out = self.eval_net(x)
        return out
