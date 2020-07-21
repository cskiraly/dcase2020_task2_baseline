"""
 @file   02_convert.py
 @brief  Script to convert model to tflite
 @author Csaba Kiraly
"""

########################################################################
# import default python-library
########################################################################
import os
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import common as com
import keras_model
import tensorflow as tf
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

        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_quant_model = converter.convert()

        print("============== MODEL SAVE ==============")
        # Save the TF Lite model.
        tflite_file = "{model}/model_{machine_type}.tflite".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        with tf.io.gfile.GFile(tflite_file, 'wb') as f:
            f.write(tflite_model)

        tflite_quant_file = "{model}/model_{machine_type}_quant.tflite".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        with tf.io.gfile.GFile(tflite_quant_file, 'wb') as f:
            f.write(tflite_quant_model)

