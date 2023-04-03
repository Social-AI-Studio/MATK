from argparse import ArgumentParser

def add_trainer_args(parser: ArgumentParser):
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--devices", type=int, default="1")

def add_train_args(parser: ArgumentParser):
    parser.add_argument("--seed", type=int, default=1001)
    
    parser.add_argument("--dataset_name", choices=["fhm", "fhm_finegrained"], default="fhm")

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--shuffle_train", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulate_gradients", type=int, default=1)

    parser.add_argument("--early_stopping", action='store_true')
    parser.add_argument("--patience", type=int, default=3, help="num. of iterations observed before early stopping")

def add_test_args(parser: ArgumentParser):
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--saved_model_filepath", default=None)