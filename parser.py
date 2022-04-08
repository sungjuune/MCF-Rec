import argparse
import string

def parse_args():
    parser = argparse.ArgumentParser(description='SJ')

    parser.add_argument("--dataset", default="naver_toy", help="dataset to train")
    parser.add_argument("--dim", default="32", type=int, help="initial embedding dimension")
    parser.add_argument("--n_layers", default=2, type=int, help="number of propagation layers")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--decay", type=float, default=1e-5, help="decay weight")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epoch", type=int, default=5000, help="epoch")
    parser.add_argument("--batch_size", type=int, default=512, help="training batch size")
    parser.add_argument('--Ks',  default='[20, 40, 60, 80, 100]', help='Metric @ K')
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="")
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')

    return parser.parse_args()