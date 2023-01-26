import argparse

from . import TrainOptions, create_dataset, generate, train, view_audio


def main() -> None:
    parser = argparse.ArgumentParser("MusicGAN")

    sub_parser = parser.add_subparsers()
    sub_parser.required = True
    sub_parser.dest = "mode"

    # Create dataset args
    create_dataset_parser = sub_parser.add_parser("create_dataset")

    create_dataset_parser.add_argument(
        "audio_path",
        type=str,
        help="can be /path/to/*.wav",
    )
    create_dataset_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="The folder where the tensor files will be saved",
    )

    # Train args
    train_parser = sub_parser.add_parser("train")

    train_parser.add_argument(
        "run",
        type=str,
        metavar="RUN_NAME",
    )

    train_parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        required=True,
    )

    train_parser.add_argument(
        "-i",
        "--input-dataset",
        required=True,
        type=str,
    )

    train_parser.add_argument(
        "--rand-channels",
        type=int,
        default=64,
    )

    train_parser.add_argument(
        "--disc-lr",
        type=float,
        default=1e-4,
    )

    train_parser.add_argument(
        "--gen-lr",
        type=float,
        default=1e-4,
    )

    train_parser.add_argument(
        "--disc-betas",
        type=float,
        nargs=2,
        action="append",
        default=[0.0, 0.9],
    )

    train_parser.add_argument(
        "--gen-betas",
        type=float,
        nargs=2,
        action="append",
        default=[0.0, 0.9],
    )

    train_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1000,
    )

    train_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
    )

    train_parser.add_argument(
        "--train-gen-every",
        type=int,
        default=5,
    )

    train_parser.add_argument(
        "--fadein-lengths",
        type=int,
        nargs="+",
        default=[
            1,
            64000,
            64000,
            64000,
            64000,
            64000,
            64000,
            64000,
        ],
    )

    train_parser.add_argument(
        "--train-lengths",
        type=int,
        nargs="+",
        default=[
            64000,
            128000,
            128000,
            128000,
            128000,
            128000,
            128000,
        ],
    )

    train_parser.add_argument(
        "--save-every",
        type=int,
        default=4000,
    )

    train_parser.add_argument(
        "--log-metrics-every",
        type=int,
        default=500,
    )

    # Generate args
    generate_parser = sub_parser.add_parser("generate")

    generate_parser.add_argument(
        "gen_dict_state",
        type=str,
    )

    generate_parser.add_argument(
        "rand_channels",
        type=int,
    )

    generate_parser.add_argument(
        "-n",
        "--nb-vec",
        type=int,
        default=10,
    )

    generate_parser.add_argument(
        "-m",
        "--nb-music",
        type=int,
        default=5,
    )

    generate_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
    )

    # View audio args
    view_audio_parser = sub_parser.add_parser("view_audio")

    view_audio_parser.add_argument(
        "-i",
        "--input-audio",
        type=str,
        required=True,
    )

    view_audio_parser.add_argument(
        "--frame-idx",
        type=int,
        required=True,
        nargs="+",
    )

    view_audio_parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    match args.mode:
        case "create_dataset":
            create_dataset(args.audio_path, args.output_dir)
        case "train":
            train(
                TrainOptions(
                    run_name=args.run,
                    dataset_path=args.input_dataset,
                    output_dir=args.out_path,
                    rand_channels=args.rand_channels,
                    disc_lr=args.disc_lr,
                    gen_lr=args.gen_lr,
                    disc_betas=args.disc_betas,
                    gen_betas=args.gen_betas,
                    nb_epoch=args.epochs,
                    batch_size=args.batch_size,
                    train_gen_every=args.train_gen_every,
                    fadein_lengths=args.fadein_lengths,
                    train_lengths=args.train_lengths,
                    save_every=args.save_every,
                    log_metrics_every=args.log_metrics_every,
                )
            )
        case "generate":
            generate(
                args.output_dir,
                args.rand_channels,
                args.gen_dict_state,
                args.nb_vec,
                args.nb_music,
            )
        case "view_audio":
            view_audio(args.input_audio, args.frame_idx, args.output_path)
        case _:
            parser.error(f"Unrecognized mode : '{args.mode}'")


if __name__ == "__main__":
    main()
