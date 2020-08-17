from src.start_pipeline import start


if __name__ == '__main__':

    dataset_path = r"C:\Users\barte\Documents\Projects\HateSpeechModels\data\APG-online-hate-classifier.csv"
    root_save_path = r"C:\Users\barte\Documents\Projects\HateSpeechModels\output"


    start(dataset_path=dataset_path,
          root_save_path=root_save_path)