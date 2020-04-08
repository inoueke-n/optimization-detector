from learning import *

DIR = "/home/davide/Desktop/flag_detection"
FEATURES = 2048


def write_dataset(model_dir, function_grained=True):
    x0, x1 = read_and_clean(DIR, function_grained)
    train, test = gen_train_test(x0, x1, function_grained, cut=0.7,
                                 chunk_size=FEATURES)
    write_binary(train, model_dir + "train.bin", function_grained,
                 chunk_size=FEATURES)
    write_binary(test, model_dir + "test.bin", function_grained,
                 chunk_size=FEATURES)


if __name__ == "__main__":
    # execute only if run as a script
    function_grained = False
    MODEL_DIR = "/home/davide/Desktop/flag_detection/models/binary_cnn_whole/"
    # write_dataset(MODEL_DIR, function_grained)
    # train = read_binary(MODEL_DIR + "train.bin", function_grained)
    # test = read_binary(MODEL_DIR + "test.bin", function_grained)
    # (X_train, y_train) = generate_sequences(train)
    # (X_test, y_test) = generate_sequences(test)
    ###########################################################################
    # # FUNCTION GRAINED
    # binary_plain_LSTM(X_train, y_train, X_test, y_test,
    #                   model_path=MODEL_DIR + "model.hdf5",
    #                   embedding_size=65536, pad_length=1024)
    # binary_convolutional_LSTM(X_train, y_train, X_test, y_test,
    #                           model_path=MODEL_DIR + "model.hdf5",
    #                   embedding_size=65536, pad_length=FEATURES)
    ###########################################################################
    # # EXECUTABLE GRAINED
    # binary_convolutional_LSTM(X_train, y_train, X_test, y_test,
    #                           model_path=MODEL_DIR + "model.hdf5",
    #                   embedding_size=256, pad_length=FEATURES)
    ###########################################################################
    # EVALUATION
    cut = 10
    while cut < 150:
        print(f"Evaluating {cut}")
        test = read_binary(MODEL_DIR + "test.bin", function_grained)
        for sample in test:
            sample["x"] = sample["x"][:cut]
        (X_test, y_test) = generate_sequences(test)
        matrix = evaluate_nn(MODEL_DIR + "model.hdf5", X_test, y_test,
                             pad_length=FEATURES)
        matrix = np.asarray(matrix)
        with open("/home/davide/Desktop/flag_detection/models/confusion_binary_cnn_whole.csv", "a") as f:
            f.write(str(cut) + ",")
            f.write(str(matrix[0][0]) + ",")
            f.write(str(matrix[0][1]) + ",")
            f.write(str(matrix[1][0]) + ",")
            f.write(str(matrix[1][1]) + "\n")
        cut = cut + 10
