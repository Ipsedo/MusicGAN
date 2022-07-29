import argparse

from . import (
    create_dataset,
    train,
    generate,
    view_audio,
)


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
        help="can be /path/to/*.wav"
    )
    create_dataset_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="The folder where the tensor files will be saved"
    )

    # Train args
    train_parser = sub_parser.add_parser("train")

    train_parser.add_argument(
        "run",
        type=str,
        metavar="RUN_NAME"
    )

    train_parser.add_argument(
        "-o", "--out-path",
        dest="out_path",
        type=str,
        required=True
    )

    train_parser.add_argument(
        "-i", "--input-dataset",
        dest="input_dataset",
        required=True,
        type=str
    )

    # Generate args
    generate_parser = sub_parser.add_parser("generate")

    generate_parser.add_argument(
        "gen_dict_state", type=str
    )

    generate_parser.add_argument(
        "rand_channels", type=int
    )

    generate_parser.add_argument(
        "-n", "--nb-vec",
        type=int, default=10
    )

    generate_parser.add_argument(
        "-m", "--nb-music",
        type=int, default=5
    )

    generate_parser.add_argument(
        "-o", "--output-dir",
        type=str, required=True
    )

    # View audio args
    view_audio_parser = sub_parser.add_parser("view_audio")

    view_audio_parser.add_argument(
        "-i", "--input-audio",
        type=str, required=True
    )

    view_audio_parser.add_argument(
        "--frame-idx",
        type=int, required=True, nargs="+"
    )

    view_audio_parser.add_argument(
        "-o", "--output-path",
        type=str, required=True
    )

    args = parser.parse_args()

    if args.mode == "create_dataset":
        create_dataset(
            args.audio_path,
            args.output_dir
        )
    elif args.mode == "train":
        train(
            args.run,
            args.input_dataset,
            args.out_path
        )
    elif args.mode == "generate":
        generate(
            args.output_dir,
            args.rand_channels,
            args.gen_dict_state,
            args.nb_vec,
            args.nb_music
        )
    elif args.mode == "view_audio":
        view_audio(
            args.input_audio,
            args.frame_idx,
            args.output_path
        )


if __name__ == '__main__':
    main()
