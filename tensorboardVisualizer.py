import warnings
warnings.filterwarnings('ignore')

import sys, os
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), 100))

    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            if word == '':
                print("Emply Line, should replaced by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))

if __name__ == "__main__":
    try:
        model_path = sys.argv[1]
        output_path  = sys.argv[2]
    except:
        print("Run as:\npython tensorboardVisualizer.py <Saved word2vec.model> <Log Folder, create if it does not exist>")
        sys.exit(0)
    model = Word2Vec.load(model_path)
    visualize(model, output_path)