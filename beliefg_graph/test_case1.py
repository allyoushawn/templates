#!/usr/bin/env python3
'''
2 var. nodes connected with 1 factor node
'''

class VarNode():
    def __init__(self, init_prior):
        self.prior = init_prior
        self.N = 100

    def set_target(v):
        self.target = v


class FactorNode():
    def __init(self, var_nodes):
        self.var_nodes = var_nodes


def update_prior(v, new_k):
    acc_prob = 0.
    for k in v.prior.keys():
        acc_prob += v.prior[k]
    other_prob = 1 - acc_prob
    point_wise_other_prob = other_prob / (v.N - len(v.prior.keys()))
    for k in new_k:
        v.prior[k] = point_wise_other_prob


def sync_priors(v1, v2):
     # v2: observation, is static, no need to touch it
     l1 = []
     for k in v2.prior.keys():
        if k not in v1.prior.keys():
            l1.append(k)
     update_prior(v1, l1)
     update_prior(v2, l2)



def main():
    v1 = VarNode({'Bill':0.6})
    o1 = VarNode({'Fred': 0.85})
    sync_priors(v1, o1)
    f1 = FactorNode([v1, o1])
    v1.set_target(f1)
    o1.set_target(f1)


    # var nodes send msg to fac nodes, but here only 1 factor node
    f1.incoming = []
    f1.incoming.append(v1.prior.copy())
    f1.incoming.append(o1.prior.copy())




if __name__ == '__main__':
    main()
