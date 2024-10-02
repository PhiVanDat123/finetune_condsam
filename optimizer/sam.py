import torch
from optimizer.optimizer_utils import enable_running_stats, disable_running_stats

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def assign_name(self, model):
        self.param_by_name = {}
        for n, p in model.named_parameters():
            self.state[p]["name"] = n
            self.param_by_name[n] = p

    @torch.no_grad()
    def first_step(self, zero_grad=False, debug=False):
        grad_norm = self._grad_norm()
        if debug:
            model_norm = self._model_norm()
        e1_len = []
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                # self.state[p]["old_p"] = p.data.clone()
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                s = (p if ("name" in self.state[p] and "weight" in self.state[p]["name"] and group["adaptive"]) else torch.tensor(1.0).to(p))
                e_w = torch.pow(s, 2) * p.grad* scale.to(p)
                if debug:
                    e1_len.append(e_w.norm())
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if zero_grad: self.zero_grad(set_to_none=False)
        if debug:
            print("\nModel-norm: %5.3f\tGrad-norm: %5.3f\tratio: %5.3f" % (
            model_norm.item(), grad_norm.item(), (grad_norm / model_norm).item()))

            c, b = torch.histogram(torch.tensor(e1_len), bins=10)
            print("e_w1_len: c :{}\n\tb: {}".format(c, b))
            
    @torch.no_grad()
    def second_step(self, zero_grad=False, debug=False, merge_grad=False, **kwargs):
        for group in self.param_groups:
            for p in group["params"]:
                # p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self, adaptive=False, by=None):
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
                    norm_object = p.grad
                if norm_object is None: continue
                norm.append(((torch.abs(p) + 0.01  if ("name" in self.state[p] and  "weight" in self.state[p]["name"] and is_adaptive) else 1.0) * norm_object).norm(p=2).to(shared_device))
        norm = torch.norm(torch.stack(norm), p=2)
        return norm
    
    # def _grad_norm(self, by=None):
    #     shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    #     norm = torch.norm(
    #                 torch.stack([
    #                     ((torch.abs(p) + 0.01 if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
    #                     for group in self.param_groups for p in group["params"]
    #                     if p.grad is not None
    #                 ]),
    #                 p=2
    #            )
    #     return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
    
    def _model_norm(self, by=None):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device

        def check(p):
            if p.grad is None:
                return False
            # if self.ignore_sigma and "rho" in self.state[p]['name']:
            #     return False
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

class FriendlySAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, sigma=1, lmbda=0.9, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FriendlySAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda
        print ('FriendlySAM sigma:', self.sigma, 'lambda:', self.lmbda)

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group["params"]:      
                if p.grad is None: continue       
                grad = p.grad.clone()
                if not "momentum" in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)
            
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    # def _grad_norm(self):
    #     shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    #     norm = torch.norm(
    #                 torch.stack([
    #                     ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
    #                     for group in self.param_groups for p in group["params"]
    #                     if p.grad is not None
    #                 ]),
    #                 p=2
    #            )
    #     return norm
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups       