from src.start_pipeline import start as start_training
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to .csv file that has columns: 'Comment_text' and 'Hateful_or_not' "
    )

    parser.add_argument(
        "--root_save_path",
        type=str,
        required=True,
        help='Folder where all the results will be saved. Inside this folder will be created folder for each run.'
    )


    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Key in 'model_map' variable in 'start_pipeline.py' file."
    )

    parser.add_argument(
        "--glove_txt_path",
        type=str,
        required=False,
        default=None,
        help="If you are trainig GLOVE model, you need to download pretrained embeddings from https://nlp.stanford.edu/projects/glove/"

    )


    parser.add_argument(
        "--max_items",
        type=int,
        required=False,
        default=None,
        help="Preprocessing all data in .csv file might take some time. If you want to just check something quickly and don't want to wait, just set some low number like 600. Only this number of rows will be used."
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help='number of epochs'
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

    print('[INFO] everything done!')

if __name__ == '__main__':
    main()
