import keras
from keras.layers import StringLookup
from keras import ops
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from config import CONFIG
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow Quickstart")

np.random.seed(42)
keras.utils.set_random_seed(42)
import logging
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



def data_splitting(base_path):
    try:
        logger.info("data split started")
        words_list = []
        words = open(f"{base_path}/words.txt", "r").readlines()
        for line in words:
            if line[0] == "#":
                continue
            if line.split(" ")[1] != "err": 
                words_list.append(line)

        print(len(words_list))
        np.random.shuffle(words_list)
        split_idx = int(0.9 * len(words_list))
        train_samples = words_list[:split_idx]
        test_samples = words_list[split_idx:]

        val_split_idx = int(0.5 * len(test_samples))
        validation_samples = test_samples[:val_split_idx]
        test_samples = test_samples[val_split_idx:]

        assert len(words_list) == len(train_samples) + len(validation_samples) + len(
            test_samples
        )

        logger.info(f"Total training samples: {len(train_samples)}")
        logger.info(f"Total validation samples: {len(validation_samples)}")
        logger.info(f"Total test samples: {len(test_samples)}")

        return train_samples,test_samples,validation_samples
    
    except Exception as e:
        logger.error(f"got this error {e}")



def get_image_paths_and_labels(samples,base_path):
    try:
        logger.info("into get_image_paths_and_labels")
        base_image_path = os.path.join(base_path, "words")
        paths = []
        corrected_samples = []
        for i, file_line in enumerate(samples):
            line_split = file_line.strip()
            line_split = line_split.split(" ")

            image_name = line_split[0]
            partI = image_name.split("-")[0]
            partII = image_name.split("-")[1]
            img_path = os.path.join(
                base_image_path, partI, partI + "-" + partII, image_name + ".png"
            )
            if os.path.getsize(img_path):
                paths.append(img_path)
                corrected_samples.append(file_line.split("\n")[0])

        return paths, corrected_samples
    except Exception as e:
        logger.error({e})


def raw_clean_labels(train_labels):
    try:
        logger.info("starting the clearn labels")
        train_labels_cleaned = []
        characters = set()
        max_len = 0

        for label in train_labels:
            label = label.split(" ")[-1].strip()
            for char in label:
                characters.add(char)

            max_len = max(max_len, len(label))
            train_labels_cleaned.append(label)

        characters = sorted(list(characters))

        logger.info(f"Maximum length: , {max_len}")
        logger.info("Vocab size: , {len(characters)}")

        return train_labels_cleaned,characters,max_len

    except Exception as e:
        logger.error(e)


def clean_labels(labels):
    try:
        logger.info("clear label just stated")
        cleaned_labels = []
        for label in labels:
            label = label.split(" ")[-1].strip()
            cleaned_labels.append(label)
        return cleaned_labels
    except Exception as e:
        logger.error(e)




def distortion_free_resize(image, img_size):
    try:
        logging.info("distortion_free_resize just started")
        w, h = img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        pad_height = h - ops.shape(image)[0]
        pad_width = w - ops.shape(image)[1]

        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )

        image = ops.transpose(image, (1, 0, 2))
        image = tf.image.flip_left_right(image)
        return image

    except Exception as e:
        logger.error(e)


batch_size = 64
padding_token = 99
image_width = 128
image_height = 32


def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = ops.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = ops.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)





class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model():
    try:
        logger.info("build_model just started")
        # Inputs to the model
        input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
        labels = keras.layers.Input(name="label", shape=(None,))

        # First conv block.
        x = keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block.
        x = keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        new_shape = ((image_width // 4), (image_height // 4) * 64)
        x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
        x = keras.layers.Dropout(0.2)(x)

        # RNNs.
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
        )(x)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
        )(x)


        x = keras.layers.Dense(
            len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
        )(x)

        # Add CTC layer for calculating CTC loss at each step.
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model.
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
        )
        # Optimizer.
        opt = keras.optimizers.Adam()
        # Compile the model and return.
        model.compile(optimizer=opt)
        return model
    except Exception as e:
        logger.error(e)

def calculate_edit_distance(labels, predictions):
    try:
        logger.info("calculate_edit_distance just started")
    
        saprse_labels = ops.cast(tf.sparse.from_dense(labels), dtype=tf.int64)
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
        predictions_decoded = keras.ops.nn.ctc_decode(
            predictions, sequence_lengths=input_len
        )[0][0][:, :max_len]
        sparse_predictions = ops.cast(
            tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
        )

        edit_distances = tf.edit_distance(
            sparse_predictions, saprse_labels, normalize=False
        )
        return tf.reduce_mean(edit_distances)
    except Exception as e:
        logger.error(e)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )


if __name__ == "__main__":

    base_path = CONFIG['data']['base_path']
    AUTOTUNE = tf.data.AUTOTUNE

    train_samples,test_samples,validation_samples = data_splitting(base_path)
    train_img_paths, train_labels =  get_image_paths_and_labels(train_samples,base_path)
    validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples,base_path)
    test_img_paths, test_labels = get_image_paths_and_labels(test_samples,base_path)

    train_labels_cleaned,characters,max_len = raw_clean_labels(train_labels)

    validation_labels_cleaned = clean_labels(validation_labels)
    test_labels_cleaned = clean_labels(test_labels)

    

    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
    validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)
    test_ds = prepare_dataset(test_img_paths, test_labels_cleaned)



    for data in train_ds.take(1):
        images, labels = data["image"], data["label"]

    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    for i in range(16):
        img = images[i]
        img = tf.image.flip_left_right(img)
        img = ops.transpose(img, (1, 0, 2))
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        # Gather indices where label!= padding_token.
        label = labels[i]
        indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
        # Convert to string.
        label = tf.strings.reduce_join(num_to_char(indices))
        label = label.numpy().decode("utf-8")

        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")

    plt.show()


  
    

    validation_images = []
    validation_labels = []

    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])


    epochs = CONFIG['data']['epoch']  # To get good results this should be at least 50.

    model = build_model()
    model.summary()
    prediction_model = keras.models.Model(
        model.get_layer(name="image").output, model.get_layer(name="dense2").output
    )
    edit_distance_callback = EditDistanceCallback(prediction_model)

    # Train the model.
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback],
)


