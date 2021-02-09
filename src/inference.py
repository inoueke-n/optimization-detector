import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras_preprocessing import sequence
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from src.extractor import extract_dot_text


def inference_file(input_file: str, model: tf.keras.Model, batch_size: int,
                   features: int) -> List[Tuple[int, int]]:
    """
    Performs the inference over a binary file with the given model. The data
    from the binary file will be extracted from the .text section, and split in
    chunks of `features` length, pre-padded with zeroes if the length is not
    sufficient. `batch_size` chunks will be fed to the model and the
    predictions collected.
    :param input_file: Path to the file from which the data will be extracted.
    :param model: The model used for inference.
    :param batch_size: Number of batches to be fed to the model.
    :param features: Number of features for each sample.
    :return: A list for tuples, where each tuple contain the sample length
    and the prediction.
    """
    data = extract_dot_text(input_file)
    if data is None:
        return []
    buffer = []
    result = []
    while len(data) > 0:
        buffer.append(data[:features])
        data = data[features:]
    while len(buffer) > 0:
        input = [np.asarray(val) for val in buffer[:batch_size]]
        shapes = [sample.shape[0] for sample in input]
        input = sequence.pad_sequences(input, maxlen=features,
                                       padding="pre", truncating="pre")
        prediction = model.predict(input, verbose=0, batch_size=batch_size)
        predicted_class = np.argmax(prediction, axis=1)
        for i in range(0, len(predicted_class)):
            result.append((shapes[i], predicted_class[i]))
        buffer = buffer[batch_size:]
    return result


def run_inference(input_files: List[str], input_dir: str, model_path: str, output: str, bs: int,
                  features: int):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    total_jobs = 0
    jobs = []
    if input_files is not None:
        for result in input_files:
            total_jobs += 1
            jobs.append(result)
    if input_dir is not None:
        files = os.listdir(input_dir)
        for result in files:
            if result.endswith(".bin"):
                jobs.append(os.path.join(input_dir, result))
                total_jobs += 1
    results = []
    model = load_model(model_path)
    progress = tqdm(total=total_jobs)
    for job in jobs:
        result = inference_file(job, model, bs, features)
        progress.update(1)
        results.append((job, result))
    progress.close()
    if output is not None:
        if os.path.exists(output):
            fp = open(output, "at")
        else:
            fp = open(output, "wt")
            fp.write("file,chunk,prediction\n")
        for result in results:
            for sample in result[1]:
                fp.write(f"\"{result[0]}\",{sample[0]},{sample[1]}\n")
        fp.close()
    else:
        for result in results:
            avg = 0
            count = 0
            for sample in result[1]:
                count += sample[0]
                avg += sample[0] * sample[1]
            print(f"\"{result[0]}\",{avg / count}\n")
