import copy



def get_detection_data_mock(folder_where_detections_files_will_be_store):
    """
    Mock of detections made by your model. They have to return a list or generator that fits function src.object_detection.prepare_gt_det_folders.prepare_detection_folder

    :return:
    """

    prediction_1 = [
        folder_where_detections_files_will_be_store, # where to save file, the same for each file, so i have realised that that's poor architectural decision
        "image_1.txt", #name of txt file where data will be stored, MUST HAVE RESPECTIVE GROUND TRUTH FILE OF THE SAME NAME
        ['person', 'car'], #list of classes on a picture
        [0.5, 0.5], #confidence of each prediction
        [ #list of bounding boxes
            [393, 122, 525, 310], # bounding boxes for person
            [313, 500, 323, 519] #bboxes for car
        ]

    ]


    prediction_2 = [
        folder_where_detections_files_will_be_store,
        # where to save file, the same for each file, so i have realised that that's poor architectural decision
        "image_2.txt",
        # name of txt file where data will be stored, MUST HAVE RESPECTIVE GROUND TRUTH FILE OF THE SAME NAME
        ['person', 'car'],  # list of classes on a picture
        [0.5, 0.5],  # confidence of each prediction
        [  # list of bounding boxes
            [0, 0, 0, 0],  # bounding boxes for person
            [1, 1, 1, 1]  # bboxes for car
        ]

    ]


    images = []

    for i in range(20):
        if i % 2 ==0:
            new_prediction = copy.deepcopy(prediction_1)
            new_prediction[1] = f"image_{i}.txt"
            images.append(new_prediction)
        else:
            new_prediction = copy.deepcopy(prediction_2)
            new_prediction[1] = f"image_{i}.txt"
            images.append(new_prediction)


    return images

