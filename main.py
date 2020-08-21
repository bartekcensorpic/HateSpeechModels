from src.start_pipeline import start as start_training
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--root_save_path",
        type=str,
        required=True
    )


    parser.add_argument(
        "--model_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--glove_txt_path",
        type=str,
        required=False,
        default=None

    )


    parser.add_argument(
        "--max_items",
        type=int,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
    )


    args = parser.parse_args()
    print(print(args))
    init(args)


def init(args):

    dataset_path = args.dataset_path #r"C:\Users\barte\Documents\Projects\HateSpeechModels\data\APG-online-hate-classifier.csv"
    root_save_path = args.root_save_path #r"C:\Users\barte\Documents\Projects\HateSpeechModels\output"
    max_items =  args.max_items
    num_epochs = args.num_epochs
    model_name = args.model_name
    glove_txt_path = args.glove_txt_path


    start_training(
        dataset_path=dataset_path,
        root_save_path=root_save_path,
        max_items=max_items,
        model_name=model_name,
        num_epochs = num_epochs,
        glove_txt_path= glove_txt_path
    )

if __name__ == '__main__':
    main()
