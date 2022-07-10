from torchvision import models
import torch
from fedscale.core.architecture_manager import Architecture_Manager
from fedscale.core.net2net import *

def widen_whole_model(torch_model, ratio: int = 1):
    dummy_input = torch.randn(10, 3, 32, 32)
    manager = Architecture_Manager(dummy_input, 'test.onnx')
    manager.parse_model(torch_model)
    dag = manager.dag

    def stripe_weight(name: str):
        if '.weight' in name:
            name = name.replace('.weight', '')
        return name

    def dfs(dag, entry):
        visited = []
        stack = [entry]
        tails = []
        inodes = []
        while len(stack) != 0:
            query = stack[-1]
            visited.append(query)
            stack.pop(-1)
            neighbors = []
            for node_id in dag.neighbors(query):
                neighbors.append(node_id)
            if len(neighbors ) == 0:
                if len(dag.nodes[query]['attr']['param_shapes']) != 0:
                    tails.append(dag.nodes[query]['attr']['name'][0])
            else:
                for neib in neighbors:
                    if neib not in visited:
                        stack.append(neib)
                if len(dag.nodes[query]['attr']['param_shapes']) != 0 and query != entry:
                    inodes.append(dag.nodes[query]['attr']['name'][0])
        tails = [stripe_weight(tail) for tail in tails]
        inodes = [stripe_weight(inode) for inode in inodes]
        return tails, inodes   

    tails, inodes = dfs(dag, manager.entry_idx)
    # print(tails, inodes)
    head = [stripe_weight(dag.nodes[manager.entry_idx]['attr']['name'][0])]
    # print(head)
    parents = head + inodes
    children = inodes + tails
    print(head, inodes, tails)
    for parent in parents:
        parent_layer = get_model_layer(torch_model, parent)
        parent_param = parent_layer.state_dict()
        out_dim = parent_param['weight'].shape[0]
        mapping = list(range(out_dim)) * ratio
        if 'bn' in parent or 'running_mean' in parent_param:
            new_parent_param = widen_batch(parent_param, mapping)
            new_parent_layer = torch.nn.BatchNorm2d(
                num_features = len(mapping),
                eps = parent_layer.eps,
                momentum = parent_layer.momentum,
                affine = parent_layer.affine,
                track_running_stats = parent_layer.track_running_stats
            )
        else:
            new_parent_param = widen_parnet_conv(parent_param, mapping)
            new_parent_layer = torch.nn.Conv2d(
                parent_layer.in_channels,
                len(mapping),
                parent_layer.kernel_size,
                stride = parent_layer.stride,
                padding = parent_layer.padding,
                groups = parent_layer.groups,
                bias = True if parent_layer.bias is not None else False
            )
        new_parent_layer.load_state_dict(new_parent_param)
        set_model_layer(torch_model, new_parent_layer, parent)
    for child in children:
        if 'bn' in child:
            continue
        child_layer = get_model_layer(torch_model, child)
        child_param = child_layer.state_dict()
        if 'running_mean' in child_param:
            continue
        in_dim = child_param['weight'].shape[1]
        mapping = list(range(in_dim)) * ratio
        if 'fc' in child:
            new_child_param = widen_child_fc(child_param, mapping)
            new_child_layer = torch.nn.Linear(
                len(mapping), child_layer.out_features,
                bias = True if child_layer.bias is not None else False,
            )
        else:
            print(child)
            new_child_param = widen_child_conv(child_param, mapping)
            new_child_layer = torch.nn.Conv2d(
                len(mapping), child_layer.out_channels,
                child_layer.kernel_size,
                stride = child_layer.stride,
                padding = child_layer.padding,
                groups = child_layer.groups,
                bias = True if child_layer.bias is not None else False
            )
        print(child)
        new_child_layer.load_state_dict(new_child_param)
        set_model_layer(torch_model, new_child_layer, child)
    return torch_model

def init_resnet18(ratio: int = 1):
    base_model = models.resnet18()
    return widen_whole_model(base_model, ratio)