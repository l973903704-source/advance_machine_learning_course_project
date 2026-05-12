from collab_common import *


if __name__ == '__main__':
    parser = build_arg_parser('neighbor')
    args = parser.parse_args()
    set_seed(args.seed)

    trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max, ex = load_dataset(args.dname, args.part)
    actual_part = len(lens)
    model = NeighborSTAN(
        t_dim=hours + 1,
        l_dim=l_max + 1,
        u_dim=u_max + 1,
        embed_dim=args.embed_dim,
        ex=ex,
        distance_matrix=mat2s,
        collab_weight=args.collab_weight,
        top_k=args.top_k,
        temperature=args.neighbor_temperature,
        momentum=args.momentum,
        c_clip=args.c_clip,
    )

    if args.load_ckpt and os.path.exists(f'{args.save_prefix}_{args.dname}.pth'):
        checkpoint = torch.load(f'{args.save_prefix}_{args.dname}.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': []}

    trainer = CollaborativeTrainer(
        model=model,
        record=records,
        trajs=trajs,
        mat1=mat1,
        mat2s=mat2s,
        mat2t=mat2t,
        labels=labels,
        lens=lens,
        dname=args.dname,
        part=actual_part,
        load_flag=args.load_ckpt,
        num_neg=args.num_neg,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epoch=args.epochs,
        aux_weight=0.0,
        weight_decay=args.weight_decay,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        best_metric=args.best_metric,
        early_stop_patience=args.early_stop_patience,
        log_c_every=args.log_c_every,
        warn_c_abs=args.warn_c_abs,
        warn_c_ratio=args.warn_c_ratio,
    )
    trainer.train(ckpt_prefix=args.save_prefix)
