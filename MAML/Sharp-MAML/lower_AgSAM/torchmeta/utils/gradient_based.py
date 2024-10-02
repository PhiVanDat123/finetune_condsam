import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchmeta.modules import MetaModule

def gradient_update_parameters(model,
                               loss,
                               train_input,
                               train_target,
                               params=None,
                               step_size=0.5,
                               first_order=False, adaptive=False, alpha=0.0005, sam_lower=True,
                               val_input=None, val_target=None, is_training=False,
                               grads_val=None):

    # print("\n\n\nHEELLOOO")


    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    key_list = params.keys()
    items_list = params.values()

    grads = torch.autograd.grad(loss,
                                params.values(),
                                # create_graph=not first_order)
                                create_graph=False)

    # print("\n\n\ntest: ", sam_lower)
    # print("\ntesta: ", alpha)
    # print("\ntestc: ", adaptive)

    l = list(range(len(grads)))
    if sam_lower:
        params_list = list(params.values())
        gradnorm_t = grad_norm(params_list, grads, adaptive)
        scale_t = alpha / (gradnorm_t + 1e-12)

        old_p = []
        for i in l:
            old_p.append(torch.zeros_like(params_list[i]))
        # if is_training:
        if True:
            # scale_t = scale_t
            scale_t = 2*scale_t
            # train_target_unique = torch.unique(train_target)
            # n_shot = len(train_target)//len(train_target_unique)
            
            # # if n_shot > 1:
            # #     n_shot = n_shot//2
            # val_input_random = []
            # val_target_random = []
            # for idx_t in train_target_unique:
            #     idx_val_t = torch.where(val_target == idx_t)[0]
            #     perm = idx_val_t[torch.randperm(len(idx_val_t))][:n_shot]
            #     val_input_random.append(val_input[perm])
            #     val_target_random.append(val_target[perm])
            idx_rand = torch.randperm(len(val_target))[:len(train_target)]
            val_input_random = val_input[idx_rand]
            val_target_random = val_target[idx_rand]
            # import pdb; pdb.set_trace()
            params_keep = OrderedDict(zip(key_list, params_list))
            model.zero_grad()
            val_logit = model(val_input_random, params=params_keep)
            loss_val = F.cross_entropy(val_logit, val_target_random)
            model.zero_grad()
            grads_val_i = torch.autograd.grad(loss_val,
                                    params_keep.values(),
                                    # create_graph=not first_order)
                                    create_graph=False)
            # grads_val = grads_val_i
            if grads_val is None:
                grads_val = [torch.zeros_like(g) for g in grads_val_i]
            #     # grads_val = grads_val_i
            # else:
            #     grads_val = [0.9*grads_val[i] + 0.1*grads_val_i[i] for i in range(len(grads_val_i))]
            # # else:
            grads_val = [0.7*grads_val[i] + 0.3*grads_val_i[i] for i in range(len(grads_val_i))]
            
            gradnorm_v = grad_norm(params_list, grads_val, adaptive)
            scale_v = alpha / (gradnorm_v + 1e-12)

            # scale_v = torch.tensor(0)
        else:
            grads_val = grads
            scale_v = torch.tensor(0)
        
        for i in l:
            e_w = (torch.pow(params_list[i], 2) if adaptive else 1.0) * (grads[i] * scale_t.to(params_list[i])-grads_val[i]*scale_v.to(params_list[i]))
            params_list[i] = params_list[i].add(e_w)  # climb to the local maximum "w + e(w)"

        params_new = OrderedDict(zip(key_list, params_list))
        model.zero_grad()
        train_logit = model(train_input, params=params_new)
        inner_loss = F.cross_entropy(train_logit, train_target)
        model.zero_grad()
        grads_new = torch.autograd.grad(inner_loss,
                                        params_new.values(),
                                        create_graph=not first_order)
                                        # create_graph=False)

        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads_new):
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads_new):
                updated_params[name] = param - step_size * grad

    else:
        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size * grad

    return updated_params, grads_val

def grad_norm(params_list, grads, adaptive):
    shared_device = params_list[0].device  # put everything on the same device, in case of model parallelism
    l = list(range(len(grads)))
    norm = torch.norm(
                torch.stack([
                    ((torch.abs(params_list[i]) if adaptive else 1.0) * grads[i]).norm(p=2).to(shared_device)
                     for i in l
                    if grads is not None
                ]),
                p=2
           )
    return norm



