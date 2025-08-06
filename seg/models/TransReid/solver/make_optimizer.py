import torch
from model.adan import Adan

def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adan':
        optimizer = Adan(params, weight_decay=cfg.SOLVER.WEIGHT_DECAY,lr=cfg.SOLVER.BASE_LR, 
        betas=cfg.SOLVER.BETAS, eps = cfg.SOLVER.EPS, max_grad_norm=cfg.SOLVER.MAX_GRAD_NORM) #引入的Adan优化器
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center


#if args.use_adan:
#        if args.bias_decay:
#            param = model_without_ddp.parameters() 
#        else: 
#            param = param_groups
#            args.weight_decay = 0.0
#        optimizer = Adan(param, weight_decay=args.weight_decay,
#        lr=args.lr, betas=args.opt_betas, eps = args.opt_eps, max_grad_norm=args.max_grad_norm
#        )