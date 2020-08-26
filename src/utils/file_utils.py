from datetime import datetime
import os


def get_output_folder(root_output_folder, human_readable_model_name):
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_folder = f"{now}_{human_readable_model_name}"

    folder_path = os.path.join(root_output_folder, model_folder)

    return folder_path
