import argparse
import os

from global_parameters import (
    DEFAULT_DATASET_DIR,
    DEFAULT_CKPT_DIR,
    TRANSFORMERS_PATH,
    SSD_DIR,
    dataset2folder,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ivqa",
        choices=[
            "ivqa",
            "msrvtt",
            "msrvttmc",
            "msvd",
            "webvid",
            "activitynet",
            "howto100m",
            "howtovqa",
            "how2qa",
            "nextqa",
            "star",
            "tgifqa/transition",
            "tgifqa/action",
            "tgifqa/frameqa",
            "tgifqa2/transition",
            "tgifqa2/action",
            "causalvid"
        ],
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="",
        choices=["", "1", "10", "20", "50"],
        help="use a subset of the generated dataset",
    )

    # Model
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        choices=["", "qa"],
        help="qa baseline does not use the video, video baseline does not use the question",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="number of layers in the multi-modal transformer",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="number of attention heads in the multi-modal transformer",
    )
    parser.add_argument(
        "--embd_dim",
        type=int,
        default=512,
        help="multi-modal transformer and final embedding dimension",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=2048,
        help="multi-modal transformer feed-forward dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout rate in the multi-modal transformer",
    )
    parser.add_argument(
        "--sentence_dim",
        type=int,
        default=2048,
        help="sentence dimension for the differentiable bag-of-words embedding the answers",
    )
    parser.add_argument(
        "--qmax_words",
        type=int,
        default=20,
        help="maximum number of words in the question",
    )
    parser.add_argument(
        "--amax_words",
        type=int,
        default=10,
        help="maximum number of words in the answer",
    )
    parser.add_argument(
        "--max_feats",
        type=int,
        default=20,
        help="maximum number of video features considered",
    )

    # Paths
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="folder where the datasets folders are stored",
    )
    parser.add_argument(
        "--ssd_dir",
        type=str,
        default=SSD_DIR,
        help="folder with ssd storage where the HowTo100M features are stored",
    )
    parser.add_argument(
        "--checkpoint_predir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="folder to store checkpoints",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="", help="subfolder to store checkpoint"
    )
    parser.add_argument(
        "--pretrain_path", type=str, default="", help="path to pretrained checkpoint"
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        default=TRANSFORMERS_PATH,
        help="path to transformer models checkpoints",
    )

    # Train
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_val", type=int, default=2048)
    parser.add_argument(
        "--n_pair",
        type=int,
        default=32,
        help="number of clips per video to consider to train on HowToVQA69M",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--test", type=int, default=0, help="use to evaluate without training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.00005, help="initial learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--clip",
        type=float,
        default=12,
        help="gradient clipping",
    )

    # Print
    parser.add_argument(
        "--freq_display", type=int, default=3, help="number of train prints per epoch"
    )
    parser.add_argument(
        "--num_thread_reader", type=int, default=16, help="number of workers"
    )

    # Masked Language Modeling and Cross-Modal Matching parameters
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--n_negs", type=int, default=1)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--min_time", type=int, default=10)
    parser.add_argument("--min_words", type=int, default=10)

    # Demo parameters
    parser.add_argument(
        "--question_example", type=str, default="", help="demo question text"
    )
    parser.add_argument("--video_example", type=str, default="", help="demo video path")
    parser.add_argument("--port", type=int, default=8899, help="demo port")
    parser.add_argument(
        "--pretrain_path2", type=str, default="", help="second demo model"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./save_models/", help="path to save dir"
    )
    parser.add_argument(
        "--mc", type=int, default=5, help="number of multiple choices"
    )
    parser.add_argument(
        "--bnum", type=int, default=10, help="number of region proposal"
    )
    parser.add_argument(
        "--cl_loss", type=float, default=0, help="trade offf with contrastive loss"
    )
    parser.add_argument(
        "--lan", type=str, default='RoBERTa', help="BERT or RoBERTa"
    )

    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = args.bert_path
    # args.save_dir = './save_dir/'
    
    #args.save_dir = os.path.join(args.checkpoint_predir, args.checkpoint_dir)

    # multiple-choice arg
    # args.mc = 4 if args.dataset == "how2qa" else 0
    # args.mc = 5 if args.dataset == "nextqa" else 0

    # feature dimension
    args.feature_dim = 2048  # S3D:1024 app_mot:4096 #2048 RoI
    args.word_dim = 768  # DistilBERT

    # Map from dataset name to folder name

    load_path = os.path.join(args.dataset_dir, args.dataset)
    args.load_path = load_path

    if args.dataset not in ["howto100m", "howtovqa"]:  # VideoQA dataset
        args.features_path = f'../data/{args.dataset}/' #os.path.join(load_path, "s3d.pth")
        # args.features_path = f'/data/datasets/{args.dataset}/'
        args.train_csv_path = os.path.join(load_path, "train.csv")
        if args.dataset == 'tgifqa':
            args.val_csv_path = os.path.join(load_path, "test.csv")
        else:
            args.val_csv_path = os.path.join(load_path, "val.csv")
        args.test_csv_path = os.path.join(load_path, "test.csv")
        args.vocab_path = os.path.join(load_path, "vocab.json")
    else:  # Pretraining dataset
        args.features_path = os.path.join(
            args.ssd_dir, "s3d_features", "howto100m_s3d_features"
        )
        if args.dataset == "howto100m":
            args.caption_path = os.path.join(
                load_path, "caption_howto100m_sw_nointersec_norepeat.pickle"
            )
            args.train_csv_path = os.path.join(
                load_path, f"s3d_features_nointersec.csv"
            )
            args.youcook_val_path = os.path.join(
                args.dataset_dir, "YouCook2", "youcook_unpooled_val.pkl"
            )
            args.msrvtt_test_csv_path = os.path.join(
                args.dataset_dir, "MSR-VTT", "MSRVTT_JSFUSION_test.csv"
            )
            args.msrvtt_test_features_path = os.path.join(
                args.dataset_dir, "MSR-VTT", "msrvtt_test_unpooled_s3d_features.pth"
            )
        elif args.dataset == "howtovqa":
            if not args.subset:
                args.caption_path = os.path.join(load_path, "howtovqa.pkl")
                args.train_csv_path = os.path.join(load_path, "train_howtovqa.csv")
                args.val_csv_path = os.path.join(load_path, "val_howtovqa.csv")
            else:
                args.caption_path = os.path.join(
                    load_path, f"howtovqa_{args.subset}.pickle"
                )
                args.train_csv_path = os.path.join(
                    load_path, f"train_howtovqa_{args.subset}.csv"
                )
                args.val_csv_path = os.path.join(
                    load_path, f"val_howtovqa_{args.subset}.csv"
                )

    return args
