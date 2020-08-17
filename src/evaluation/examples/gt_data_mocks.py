import copy

def get_ground_truth_data_mock(folder_where_gt_files_will_be_store):
    """
    Mock of ground truth data. They have to return a list or generator that fits function src.object_detection.prepare_gt_det_folders.prepare_ground_truth_folder

    :return:
    """

    image_1 = [
        folder_where_gt_files_will_be_store,  # where to save file, the same for each file, so i have realised that that's poor architectural decision
        "image_1.txt",  #name of txt file where data will be stored
        ['person', 'car'],  #list of classes on a picture
        [ #list of bounding boxes
            [393, 122, 525, 310],  # bounding boxes for person
            [313, 500, 323, 519]  # bboxes for car
        ]
    ]

    image_2 = [
        folder_where_gt_files_will_be_store,  # where to save file, the same for each file, so i have realised that that's poor architectural decision
        "image_2.txt",  # name of txt file where data will be stored
        ['person', 'car'],  # list of classes on a picture
        [  # list of bounding boxes
            [393, 122, 525, 310],  # bounding boxes for person
            [313, 500, 323, 519]  # bboxes for car
        ]
    ]

    images = []

    for i in range(20):
        if i % 2 ==0:
            new_image = copy.deepcopy(image_1)
            new_image[1] = f"image_{i}.txt"
            images.append(new_image)
        else:
            new_image = copy.deepcopy(image_2)
            new_image[1] = f"image_{i}.txt"
            images.append(new_image)


    return images
