from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)

valid_split = 0.1
shuffle_seed = 43
sample_rate = 16000
scale = 0.5
batch_size = 128
epochs = 50
labels = ["unknown"]
# Load the saved model
model = tf.keras.models.load_model('model.h5')


#Preprocessing Functions
def paths_and_labels_to_dataset(audio_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))

def path_to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, sample_rate)
    return audio

def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        audio = audio + noise * prop * scale

    return audio

def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

def paths_to_dataset(audio_paths):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    return tf.data.Dataset.zip((path_ds))

def predict(path, labels):
    test = paths_and_labels_to_dataset(path, labels)
    test = test.shuffle(buffer_size=batch_size * 8, seed=shuffle_seed).batch(
    batch_size
    )
    test = test.prefetch(tf.data.experimental.AUTOTUNE)

    for audios, labels in test.take(1):
        ffts = audio_to_fft(audios)
        y_pred = model.predict(ffts)
        rnd = np.random.randint(0, 1, 1)
        audios = audios.numpy()[rnd, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

    for index in range(1):
            class_names = ['Nelson_Mandela', 'Jens_Stoltenberg', 'Julia_Gillard', 'Benjamin_Netanyau', 'Magaret_Tarcher']
            return class_names[y_pred[index]]



@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Receive the audio file from the POST request
        audio_file = request.files['audio']

        if audio_file:
            # Save the audio file to a temporary location
            temp_audio_path = "temp_audio.wav"
            audio_file.save(temp_audio_path)

            result = predict([temp_audio_path], labels)
            
            
            
            # Return the result as JSON
            response_data = {'result': result}
            os.remove(temp_audio_path)  # Remove the temporary files
            return jsonify(response_data)
        else:
            return jsonify({'error': 'No audio file provided in the request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
