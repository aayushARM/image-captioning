
import tensorflow as tf
import numpy as np
import flask
from flask import request, jsonify
import time
from io import BytesIO
from tensorflow.python.lib.io import file_io
import urllib.request as req
import shutil

app = flask.Flask(__name__)
gcs_public_data = 'https://storage.googleapis.com/image-captioning-196706-bucket/data/'
graph_file = 'frozen_final_graph.pb'
dict_file = 'idx_to_word.npy'

# Firstly, run all CPU, memory intensive operations only once, before starting app.
# Loaded Graph and Dictionary will then stay in RAM, always ready to serve requests.
# Doing this greatly reduces request serving time!

if not tf.gfile.Exists(graph_file):
    #Download the frozen graph from GCS bucket and save locally
    print('Downloading frozen graph...')
    with req.urlopen(gcs_public_data + graph_file) as response, tf.gfile.Open(graph_file, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print('Done.')

if not tf.gfile.Exists(dict_file):
    #Download the dictionary from GCS bucket and save locally
    with req.urlopen(gcs_public_data + dict_file) as response, tf.gfile.Open(dict_file, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

#Load graph in memory
with tf.gfile.Open(graph_file, 'rb') as f:
    fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

#Load dictionary in memory
f = BytesIO(file_io.read_file_to_string(dict_file, binary_mode=True))
idx_to_word = np.load(f).tolist()

tf.import_graph_def(graph_def)
graph = tf.get_default_graph()

#get last output tensor from graph
int_caption = graph.get_tensor_by_name('import/int_caption:0')
sess= tf.InteractiveSession()
print('Flask server started.')

# route all request to '/' to this method
@app.route('/', methods=['POST','GET'])
def run_inference():
    if request.method == 'POST':
        upload_file = request.files['file']
        # image_path = image_folder + upload_file.filename
        # upload_file.save(image_path)
        image_data = upload_file.read()
        image = tf.image.decode_jpeg(image_data, channels=3, dct_method="INTEGER_ACCURATE")
        image_dims = np.shape(sess.run(image))

        # pre-process received image, based on 3 possible cases:
        print('Image Dimensions: ', image_dims)
        if image_dims[0]<331 or image_dims[1]<331:
            image = tf.image.resize_images(image, size=[331, 331])
            print('Image case: 1')
        elif image_dims[0]<346 or image_dims[1]<346:
            image = tf.random_crop(image, [331, 331, 3])
            print('Image case: 2')
        else:
            image = tf.image.resize_images(image, size=[346, 346])
            image = tf.random_crop(image, [331, 331, 3])
            print('Image case: 3')

        image = tf.cast(image, tf.uint8) # final image shape: [331, 331, 3]
        print('Running the graph...')
        start_time = time.time()
        image = sess.run(image)

        # give image as input, get integer caption as output
        int_caption_out = sess.run(int_caption, feed_dict={'import/processed_image:0': image})

        # generate final caption using dictionary
        caption = ''
        cap_len = len(int_caption_out)
        for j in range(cap_len):
            word = idx_to_word[int_caption_out[j]]
            if (j + 1 != cap_len) and (int_caption_out[j + 1] != 2): # index of end-token = 2
                caption += word + ' '
            else:
                caption += word
                break

        # explicitly add period if not present
        if caption[-1] != '.':
            caption += ' .'
        print('Done. Request took {:.2f}s'.format(time.time() - start_time))

        return jsonify(caption)

    # When called from a Browser(GET), an interface to upload images will be returned...
    else:
        return '''
            <!doctype html>
            <html lang="en">
            <head>
            <title>Image Caption Generator</title>
            </head>
            <body>
            <h2 style="color:black">Image Caption Generator</h4>
            <div style="margin-top:3%">
                <h4 style="color:black">Upload an image to generate captions for:</h4>
                <form method=post enctype=multipart/form-data>
                    <p><input type=file name=file>
                    <input type=submit style="color:black;" value=Upload>
                </form>
            </div>
            </body>
            </html>
            '''

if __name__ == '__main__':
    # When run on 0.0.0.0, Flask automatically maps public IP of machine to current app
    app.run(host="0.0.0.0", port=int("5000"), debug=True, use_reloader=False)
