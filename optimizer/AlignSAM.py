import torch
from optimizer.optimizer_utils import *

class AlignSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, agsam_lambda=2.0, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(AlignSAM, self).__init__(params, defaults)
        
        self.agsam_lambda = agsam_lambda
        self.rho = rho

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.cosine_store = {
            "tG_vU": [],
            "tU_vU": []
        }
        self.cosine_store2 = []

    def assign_name(self, model):
        self.param_by_name = {}
        for n, p in model.named_parameters():
            self.state[p]["name"] = n
            self.param_by_name[n] = p

    @torch.no_grad()
    def grad_store(self, zero_grad=False, debug=False, grad_name="v_update", is_val=False, store_model=False):
        for group in self.param_groups:
            for p in group["params"]:
                if store_model:
                    self.state[p]["old_p"] = p.data.clone()
                if p.grad is None: continue
                self.state[p][grad_name] = p.grad.clone()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self, zero_grad=False, debug=False):
        grad_norm_t = self._grad_norm(by="grad_t")
        grad_norm_v = self._grad_norm(by="v_update")

        scale_t = self.agsam_lambda*self.rho / (grad_norm_t + 1e-12)
        scale_v = self.rho / (grad_norm_v + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"].clone()
                if "grad_t" in self.state[p]:
                    if "v_update" in self.state[p]:
                        e_w = scale_t*self.state[p]["grad_t"] - scale_v*self.state[p]["v_update"]
                    else:
                        e_w = scale_t*self.state[p]["grad_t"]
                    s = (p if "name" in self.state[p] and "weight" in self.state[p]["name"] and group["adaptive"] else torch.tensor(1.0).to(p))
                    p.add_(e_w*torch.pow(s, 2))  # climb to the local maximum "w + e(w)"
        if zero_grad: self.zero_grad()
        if debug:
            model_norm = self._model_norm()
            print("\nModel-norm: %5.3f\tGrad-norm: %5.3f\tratio: %5.3f\tGrad-norm-v: %5.3f\tratio: %5.3f" % (
            model_norm.item(), grad_norm_t.item(), (grad_norm_t / model_norm).item(), grad_norm_v.item(), (grad_norm_v / model_norm).item()))
         
    @torch.no_grad()
    def second_step(self, zero_grad=False, debug=False, merge_grad=False, **kwargs):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                if p.grad is None: continue
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()
    
    
    def _grad_norm(self, adaptive=False, by=None, one_weight=False):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = []
        for group in self.param_groups:
            is_adaptive = adaptive
            if not is_adaptive:
                is_adaptive = group["adaptive"]

            for p in group["params"]:
                norm_object = None
                if by and by in self.state[p]:
                    norm_object = self.state[p][by]
                elif by is None:
                    if one_weight and "name" in self.state[p] and  "weight" in self.state[p]["name"]:
                        norm_object = p.grad
                if norm_object is None: continue
                norm.append(((torch.abs(p) + 0.01 if "name" in self.state[p] and  "weight" in self.state[p]["name"] and is_adaptive else 1.0) * norm_object).norm(p=2).to(shared_device))
        if len(norm) > 0:
            norm = torch.norm(torch.stack(norm), p=2)
        else:
            norm = torch.tensor(0.0).to(shared_device)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
    
    def _model_norm(self, by=None):
        shared_device = self.param_groups[0]["params"][0].device

        def check(p):
            if p.grad is None:
                return False
            return True
        if by:
            norm = torch.norm(
                torch.stack([
                    self.state[p][by].norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if check(p)
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                    torch.stack([
                        p.data.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if check(p)
                    ]),
                    p=2
               )
        return norm

