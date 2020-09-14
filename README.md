It’s called Hate Speech Models but as long as it follows same structure as `data/APG-online-hate-classifier.csv` any dataset is fine. Check out `def get_dfs` function in `get_data.py` file.

##### You may pick any model that:
1. Has defined training function (see any file from `models` folder)
2. Is 'registered' in `model_map` variable in function `start` in file `start_pipeline.py`. That's a lot of starts.

To 'register' model, add appropriate function following interface, as specified in 'get_models.py' file. Each function returns training function and path to folder, where it will save outputs. 

## Example inputs:

Each parameter is explained in `main.py` file.

### Train GLOVE model

```
--dataset_path
"/home/ubuntu/Desktop/Projects/HateSpeechModels/data/APG-online-hate-classifier.csv"
--root_save_path
"/mnt/efs/hatespeech_results"
--num_epochs
10
--model_name
"glove"
--glove_txt_path
"/home/ubuntu/Desktop/Projects/glove_txts/glove.6B.100d.txt"
```

### Train distilbert model

```
--dataset_path "/home/ubuntu/Desktop/Projects/HateSpeechModels/data/APG-online-hate-classifier.csv" 
--root_save_path "/mnt/efs/hatespeech_results"
--num_epochs 8 
--model_name "distilbert_transformers_complex"
```