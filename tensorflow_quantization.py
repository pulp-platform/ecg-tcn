#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import numpy as np
import tensorflow as tf
from utils.tf_model_maker import *
from utils.tf_hex import *
import os
export_path = 'model/tf/'
X_train = np.load('data/X_train_ECG5000.npy')
y_train = np.load('data/y_train_ECG5000.npy')
X_test = np.load('data/X_test_ECG5000.npy')
y_test = np.load('data/y_test_ECG5000.npy')
X_train_1D = X_train.copy()
y_train_1D = y_train.copy()
X_test_1D = X_test.copy()
y_test_1D = y_test.copy()
X_train_1D.shape = (500,140,1)
X_test_1D.shape = (4500,140,1)
X_train_1D = X_train_1D.astype(np.float32)
X_test_1D = X_test_1D.astype(np.float32)
X_train.shape = (500,1,140,1)
X_test.shape = (4500,1,140,1)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)
X_train_1D = tf.convert_to_tensor(X_train_1D)
X_test_1D = tf.convert_to_tensor(X_test_1D)
y_train_1D = tf.convert_to_tensor(y_train_1D)
y_test_1D = tf.convert_to_tensor(y_test_1D)
model_1D = tf_model(X_train_1D)
batch_size = 30
epochs = 10
model_1D.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model_1D.fit(X_train_1D, y_train_1D, batch_size, epochs, validation_data=(X_test_1D,y_test_1D))
model_2D = tf_model_zero(X_train)
model_2D.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model_weight_setter(model_2D,model_1D)
score = model_1D.evaluate(X_test_1D, y_test_1D)
print(score)
score = model_2D.evaluate(X_test, y_test)
print(score)
os.makedirs(export_path, exist_ok=True) 
model_2D.save("model/tf/ECG-TCN.h5")
train_set = X_train.numpy()
test_set = X_test.numpy()
train_labels = y_train.numpy()
test_labels = y_test.numpy()
tflite_model_name = 'ECG-TCN'
converter = tf.lite.TFLiteConverter.from_keras_model(model_2D)
converter.experimental_new_converter =True
quantize = True
if (quantize):
    def representative_dataset():
        for i in range(100):
            yield([train_set[i].reshape(1,1,140,1)])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
open('model/tf/'+tflite_model_name + '.tflite', 'wb').write(tflite_model)
model_name = 'ECG-TCN'
with open('model/tf/'+model_name + '.h', 'w') as file:
    file.write(c_array_maker(tflite_model, model_name))
tflite_interpreter = tf.lite.Interpreter(model_path='model/tf/'+tflite_model_name + '.tflite')
tflite_interpreter.allocate_tensors()
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
predictions = np.zeros((len(test_set),), dtype=int)
input_scale, input_zero_point = input_details[0]["quantization"]
for i in range(len(test_set)):
    val_batch = test_set[i]
    val_batch = val_batch / input_scale + input_zero_point
    val_batch = np.expand_dims(val_batch, axis=0).astype(input_details[0]["dtype"])
    tflite_interpreter.set_tensor(input_details[0]['index'], val_batch)
    tflite_interpreter.allocate_tensors()
    tflite_interpreter.invoke()
    tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    predictions[i] = output.argmax()
sum = 0
for i in range(len(predictions)):
    if (predictions[i] == test_labels[i]):
        sum = sum + 1
accuracy_score = sum / 4500
print("Accuracy of quantized to int8 model is {}%".format(accuracy_score*100))
print("Compared to float32 accuracy of {}%".format(score[1]*100))
print("We have a change of {}%".format((accuracy_score-score[1])*100))
