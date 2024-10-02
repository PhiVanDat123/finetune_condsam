import torch
from optimizer.optimizer_utils import *

class AgnosticSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, rho_lst=[0.05, 0.05], agsam_mom=0.9, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(AgnosticSAM, self).__init__(params, defaults)
        
        self.rho1 = rho_lst[0]
        self.rho2 = rho_lst[1]
        self.agsam_mom = agsam_mom

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
                if is_val:
                    if grad_name not in self.state[p]:
                        self.state[p][grad_name] = torch.zeros_like(p.grad)
                    self.state[p][grad_name] = (1-self.agsam_mom)*p.grad.clone() + self.agsam_mom*self.state[p][grad_name]
                else:
                    self.state[p][grad_name] = p.grad.clone()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def val_first_step(self, zero_grad=False, debug=False, warm_up=1.0):
        grad_norm_v = self._grad_norm(by="grad_v")
        
        scale_v = self.rho2 / (grad_norm_v + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                
                if "grad_v" in self.state[p]:
                    e_w = scale_v*self.state[p]["grad_v"]
                    s = (p if "name" in self.state[p] and "weight" in self.state[p]["name"] and group["adaptive"] else torch.tensor(1.0).to(p))
                    p.add_(e_w*torch.pow(s, 2))  # climb to the local maximum "w + e(w)"
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def calculate_cosine(self, grad_name=[], debug=False):
        cosine = 0.0
        count = 0
        grad1_name = grad_name[0]
        grad2_name = grad_name[1]
        grad_1_norm = 0.0
        grad_2_norm = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if grad1_name in self.state[p]:
                    cosine += (self.state[p][grad1_name]*self.state[p][grad2_name]).sum()
                    grad_1_norm += (self.state[p][grad1_name]**2).sum()
                    grad_2_norm += (self.state[p][grad2_name]**2).sum()
                count += 1

        self.cosine_store2.append(cosine/torch.sqrt(grad_1_norm*grad_2_norm))
        if debug:
            print(grad_name, "Cosine after update: {:.3f}\tMean: {:3f}\tMean 200iters: {:.3f}".format(self.cosine_store2[-1], sum(self.cosine_store2)/len(self.cosine_store2), sum(self.cosine_store2[-20:])/len(self.cosine_store2[-20:])))

    @torch.no_grad()
    def first_step(self, zero_grad=False, debug=False, warm_up=1.0):
        grad_norm_t = self._grad_norm(by="grad_t")
        grad_norm_v = self._grad_norm(by="v_update")

        scale_t = self.rho1 / (grad_norm_t + 1e-12)
        scale_v = self.rho2 / (grad_norm_v + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"].clone()
                if "grad_t" in self.state[p]:
                    if "v_update" in self.state[p]:
                        e_w = scale_t*self.state[p]["grad_t"] - warm_up*scale_v*self.state[p]["v_update"]
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
        cosine_model = {
            "tG_vU": 0.0, 
            "tU_vU": 0.0, 
        }
        # count = 0
        grad_norm_t = 0.0
        grad_norm_v = 0.0
        grad_norm_tU = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                if p.grad is None: continue
                
                if "t_update" not in self.state[p]:
                    self.state[p]["t_update"] = torch.zeros_like(p.grad)
                    
                self.state[p]["t_update"] = p.grad.clone()
                
                cosine_model["tG_vU"] += (self.state[p]['grad_t']*self.state[p]["v_update"]).sum()
                cosine_model["tU_vU"] += (self.state[p]['t_update']*self.state[p]["v_update"]).sum()
                grad_norm_t += (self.state[p]['grad_t']**2).sum()
                grad_norm_tU += (self.state[p]['t_update']**2).sum()
                grad_norm_v += (self.state[p]["v_update"]**2).sum()
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

        cosine_model["tG_vU"] = cosine_model["tG_vU"]/(torch.sqrt(grad_norm_t*grad_norm_v))
        cosine_model["tU_vU"] = cosine_model["tU_vU"]/(torch.sqrt(grad_norm_tU*grad_norm_v))
        self.cosine_store["tG_vU"].append(cosine_model["tG_vU"].item())
        self.cosine_store["tU_vU"].append(cosine_model["tU_vU"].item())
        if debug:
            print("Cosine tG_vU: {:.3f}\tMean: {:3f}\tMean 200iters: {:.3f}".format(self.cosine_store["tG_vU"][-1], sum(self.cosine_store["tG_vU"])/len(self.cosine_store["tG_vU"]), sum(self.cosine_store["tG_vU"][-20:])/len(self.cosine_store["tG_vU"][-20:])))
            print("Cosine tU_vU: {:.3f}\tMean: {:3f}\tMean 200iters: {:.3f}".format(self.cosine_store["tU_vU"][-1], sum(self.cosine_store["tU_vU"])/len(self.cosine_store["tU_vU"]), sum(self.cosine_store["tU_vU"][-20:])/len(self.cosine_store["tU_vU"][-20:])))

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

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

