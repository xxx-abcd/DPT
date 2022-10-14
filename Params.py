import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')

    parser.add_argument('--device', default='cuda:0', type=str, help="['0','1','cpu']")
    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--opt_base_lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--opt_weight_decay', default=1e-4, type=float, help='weight decay regularizer')

    parser.add_argument('--batch', default=8192, type=int, help='batch size')
    # parser.add_argument('--gnn_layer', default='[16,16,16]', type=str, help='name of dataset: IJCAI_15, Tmall')
    parser.add_argument('--gnn_layer', default='[32,32,32]', type=str, help='name of dataset: IJCAI_15, Tmall')

    parser.add_argument('--dataset', default='Tmall', type=str, help='name of dataset: IJCAI_15, Tmall')
    parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer,')
    parser.add_argument('--opt_max_lr', default=5e-3, type=float,
                        help='learning rate Ijcai: 2e-3 Tmall:5e-3, TMALL:prompt add 2e-3')
    #

    parser.add_argument('--epoch', default=120, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
    parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
    parser.add_argument('--dim', default=32, type=int, help='embedding size:IJCAI-16, Tmall-32')
    parser.add_argument('--memosize', default=2, type=int, help='memory size')
    parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')

    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--shoot', default=10, type=int, help='K of top k')
    parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
    parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
    parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')
    parser.add_argument('--keepRate', default=0.7, type=float, help='rate for dropout')
    parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')
    parser.add_argument('--slot', default=5, type=int, help='length of time slots')
    parser.add_argument('--graphSampleN', default=25000, type=int,
                        help='use 25000 for training and 200000 for testing, empirically')
    parser.add_argument('--divSize', default=50, type=int, help='div size for smallTestEpoch')
    parser.add_argument('--isload', default=False, type=bool, help='whether load model')
    parser.add_argument('--isJustTest', default=False, type=bool, help='whether load model')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--inner_product_mult', default=1, type=float, help='multiplier for the result')
    parser.add_argument('--drop_rate', default=0.5, type=float, help='drop_rate')
    parser.add_argument('--drop_rate1', default=0.5, type=float, help='drop_rate')

    # parser.add_argument('--loadModelPath', default='/home/ww/Code/work3/BSTRec/Model/IJCAI_15/for_meta_hidden_dim_dim__8_IJCAI_15_2021_07_10__14_11_55_lr_0.0003_reg_0.001_batch_size_4096_gnn_layer_[16,16,16].pth', type=str, help='loadModelPath')

    # # pre-train
    #
    # parser.add_argument('--is_denoising', default=False , type=bool, help='whether load model')
    # parser.add_argument('--pattern', default=True , type=bool, help='whether load model')
    # parser.add_argument('--pretrain', default=False , type=bool, help='whether load model')
    # parser.add_argument('--pre_mode_2', default=True , type=bool, help='pre: L1+l2+l3 -> tune: l4')
    #
    # parser.add_argument('--prompt', default=False , type=bool, help='whether load model')
    # parser.add_argument('--deep', default=True , type=bool, help='whether load model')
    # # parser.add_argument('--flag', default='denoising_test_2', type=str, help='name of saved model')
    #
    # parser.add_argument('--pre_flag', default='prompt_demo_2', type=str, help='name of saved model')
    # parser.add_argument('--prompt_flag', default='prompt_demo_2_1', type=str, help='name of saved model')
    # parser.add_argument('--gumbel', default=0.2, type=float, help='rate for dropout')
    # parser.add_argument('--noise_lambda', default=-1, type=float, help='rate for dropout')

    # prompt : deep head
    # # prompt
    #
    #
    parser.add_argument('--pattern', default=False, type=bool)
    parser.add_argument('--prompt', default=True, type=bool, help='whether prompt/denoise')
    parser.add_argument('--denoise_tune', default=False, type=bool,
                        help='whether denoise, prompt=False,denoise_tune=False for the first stage')

    parser.add_argument('--wsdm', default=False, type=bool, help='whether leverage wsdm21 truank')

    # parser.add_argument('--pre_flag', default='prompt_demo_3_bias', type=str, help='name of saved model')  # 这个是IJCAI效果最好的，但是TMall的target behavior 没还原上, 可以先走后边的prompt看看效果
    parser.add_argument('--pre_flag', default='prompt_demo_3_bias_target_2', type=str,
                        help='name of saved model, 正常是target，试试不同的decay')
    parser.add_argument('--prompt_flag', default='prompt_demo_3_1_1', type=str, help='name of saved model')
    # parser.add_argument('--tune_flag', default='prompt_demo_3_0_gcn', type=str, help='name of saved model')  # IJCAI肯定是3_0应该是重新训练的GCN, 目前在CUDA1上正在试继承GCN 但现在试一下如果不继承之前的GCN参数会如何 3_0_gcn
    parser.add_argument('--tune_flag', default='prompt_demo_3_0', type=str, help='name of saved model')  # 目前Tmall最好的
    # parser.add_argument('--tune_flag', default='prompt_demo_3_wsdm', type=str, help='name of saved model')
    # parser.add_argument('--gumbel', default=0.001, type=float, help='IJCAI:0.49, Tmall:0.1')
    parser.add_argument('--gumbel', default=0.48, type=float, help='disturber')

    parser.add_argument('--deep', default=True, type=bool, help='whether load model')
    parser.add_argument('--vector', default=True, type=bool, help='whether load model')
    parser.add_argument('--head', default=False, type=bool, help='pre: L1+l2+l3 -> tune: l4')
    # parser.add_argument('--is_pattern_dumole', default=True , type=bool, help='pre: L1+l2+l3 -> tune: l4')
    parser.add_argument('--noise_lambda', default=0.1, type=float, help='rate for dropout')

    parser.add_argument('--just_test', default=False, type=bool, help='result evaluation performance')

    # # shallow - head
    # parser.add_argument('--deep', default=False , type=bool, help='whether load model')
    # parser.add_argument('--head', default=False , type=bool, help='pre: L1+l2+l3 -> tune: l4')
    # parser.add_argument('--noise_lambda', default=-1, type=float, help='rate for dropout')

    # # deep - head
    # parser.add_argument('--deep', default=True , type=bool, help='whether load model')
    # parser.add_argument('--head', default=False , type=bool, help='pre: L1+l2+l3 -> tune: l4')
    # parser.add_argument('--noise_lambda', default=-1, type=float, help='rate for dropout')

    #
    # # shallow + head
    # parser.add_argument('--deep', default=False , type=bool, help='whether load model')
    # parser.add_argument('--head', default=True , type=bool, help='pre: L1+l2+l3 -> tune: l4')
    # # parser.add_argument('--noise_lambda', default=0.1, type=float, help='rate for dropout')
    # parser.add_argument('--noise_lambda', default=0.5, type=float, help='rate for dropout')

    #
    # deep + head

    return parser.parse_args()


args = parse_args()
# args.user = 147894
# args.item = 99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
# args.user = 19800
# args.item = 22734


args.decay_step = 10000 // args.batch
