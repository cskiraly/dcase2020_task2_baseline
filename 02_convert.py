"""
 @file   02_convert.py
 @brief  Script to convert model to tflite
 @author Csaba Kiraly
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import common as com
import keras_model
import tensorflow as tf
import numpy
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# main 02_convert.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # set model path
        model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                machine_type=machine_type)

        # load model file
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)
        model = keras_model.load_model(model_file)
        model.summary()

        # Convert the model to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_file = "{model}/model_{machine_type}.tflite".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        with tf.io.gfile.GFile(tflite_file, 'wb') as f:
            f.write(tflite_model)

        # Quantization of weights (but not the activations)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_file = "{model}/model_{machine_type}_quant.tflite".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        with tf.io.gfile.GFile(tflite_file, 'wb') as f:
            f.write(tflite_model)

        print("============== DATASET_GENERATOR ==============")
        files = com.file_list_generator(target_dir)
        train_data = com.list_to_vector_array(files,
                                          msg="generate train_dataset",
                                          n_mels=param["feature"]["n_mels"],
                                          frames=param["feature"]["frames"],
                                          n_fft=param["feature"]["n_fft"],
                                          hop_length=param["feature"]["hop_length"],
                                          power=param["feature"]["power"])

        def representative_dataset_gen():
            for sample in train_data[::5]:
                sample = numpy.expand_dims(sample.astype(numpy.float32), axis=0)
                yield [sample]

        # Full integer quantization of weights and activations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        tflite_file = "{model}/model_{machine_type}_quant_fullint.tflite".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        with tf.io.gfile.GFile(tflite_file, 'wb') as f:
            f.write(tflite_model)


        # Full integer quantization of weights and activations for micro
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.int8  # or tf.uint8
        #converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_model = converter.convert()
        tflite_file = "{model}/model_{machine_type}_quant_fullint_micro.tflite".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        with tf.io.gfile.GFile(tflite_file, 'wb') as f:
            f.write(tflite_model)
