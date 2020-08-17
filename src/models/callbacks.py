import tensorflow as tf

from src.models.clr_callback import CyclicLR


def get_callbacks(
    best_model_checkpoint_path: str, csv_logger_path: str, tensorboard_logdir: str, metric_to_monitor='val_f1_m', checkpoint_mode = 'max'
):
    """
    Prepares callbacks for model training

    :param best_model_checkpoint_path: path where model is saved
    :param csv_logger_path: path where training CSV logs are stored
    :param tensorboard_logdir: path where tensorboard logs will be stored
    :return: learning_rate_reduction, checkpoint_callback, csv_logger_callback,tensorboard_callback
    """
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=metric_to_monitor, patience=2, verbose=1, factor=0.2, min_lr=0.00001
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        best_model_checkpoint_path,
        monitor=metric_to_monitor,
        verbose=1,
        save_best_only=True,
        mode=checkpoint_mode,
    )

    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        csv_logger_path, append=True, separator=";"
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir)

    clr = CyclicLR(base_lr=0.0001, max_lr=0.0001,
                        step_size=2000., mode='exp_range',
                        gamma=0.99994)

    return (
        learning_rate_reduction,
        checkpoint_callback,
        #csv_logger_callback,
        tensorboard_callback,
        #clr,
    )
