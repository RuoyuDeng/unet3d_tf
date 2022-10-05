import os
import numpy as np
import tensorflow as tf
import random
# from IPython import embed

def randrange(max_range):
    return 0 if max_range == 0 else random.randrange(max_range)


def get_cords(cord, idx):
    patch_size = [128, 128, 128]
    return cord[idx], cord[idx] + patch_size[idx]


def write_tfrecords(np_files_path, output_dir, crop = True, reshape = True):
    """ Convert numpy array to rfrecord

    :param np_files_path: String representing path where np files are stored
    :param output_dir: String representing path where to write
    """
    # list all files
    files = os.listdir(np_files_path)
    files.sort()

    # split files into samples list and labels list
    imgs = [os.path.join(np_files_path, file) for file in files if "_x" in file]  # list of file names (strings)
    lbls = [os.path.join(np_files_path, file) for file in files if "_y" in file]  # list of file names (strings)

    if reshape:
        # once such behavior is done, it is saved forever
        for i in range(len(imgs)):
            print("current label file:", lbls[i])
            cur_img = np.load(imgs[i])
            cur_img = np.moveaxis(cur_img, 0, -1) # 1,190,392,392 -> 190,392,392,1
            new_img = np.moveaxis(cur_img, 0, 2) # 392,392,190
            new_lbl = np.squeeze(np.load(lbls[i]), 0)
            np.save(imgs[i], new_img)
            np.save(lbls[i], new_lbl)

    if crop:
        for i in range(len(imgs)):
            img = np.load(imgs[i])
            lbl = np.load(lbls[i])
            ranges = [s - p for s, p in zip(img.shape[:-1], [128, 128, 128])]
            cord = [randrange(x) for x in ranges]
            low_x, high_x = get_cords(cord, 0)
            low_y, high_y = get_cords(cord, 1)
            low_z, high_z = get_cords(cord, 2)
            new_img = img[low_x:high_x, low_y:high_y, low_z:high_z, :]
            new_lbl = lbl[low_x:high_x, low_y:high_y, low_z:high_z]
            np.save(imgs[i], new_img)
            np.save(lbls[i], new_lbl)

    num_patients = len(imgs)
    
    img_list = []
    lbl_list = []
    mean_list = []
    std_list = []

    for i, img in enumerate(imgs):
        print("Starting case", i + 1)
        # image_array = np.load(img)  # array type
        # print(image_array.shape)
        image_array = np.load(img)
        label_array = np.load(lbls[i]).astype(np.uint8)  # array type
        mean = np.mean(image_array)
        std = np.std(image_array)
        
        
        img_list.append(image_array)
        lbl_list.append(label_array)
        mean_list.append(mean)
        std_list.append(std)
        
        # write to file
        output_filename = os.path.join(output_dir, "volume-{}.tfrecord".format(i))
        file_list = list(zip(np.array(img_list), 
                            np.array(lbl_list),
                            np.array(mean_list),
                            np.array(std_list)))

        # call np_to_tfrecords
        writer = tf.io.TFRecordWriter(output_filename)
        # embed()
        for file_item in file_list:
            # print(file_item[0].dtype)
            sample = file_item[0].flatten().tostring()
            label = file_item[1].flatten().tostring()
            mean = file_item[2].astype(np.float32).flatten()
            stdev = file_item[3].astype(np.float32).flatten()

            d_feature = {}
            d_feature['X'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample]))
            d_feature['Y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
            d_feature['mean'] = tf.train.Feature(float_list=tf.train.FloatList(value=mean))
            d_feature['stdev'] = tf.train.Feature(float_list=tf.train.FloatList(value=stdev))

            features = tf.train.Features(feature = d_feature)
            example = tf.train.Example(features = features)
            serialized = example.SerializeToString()
            writer.write(serialized)
        writer.close()

        # clear lists
        img_list = []
        lbl_list = []
        mean_list = []
        std_list = []

        print("Case", i + 1, "done")


if __name__ == "__main__":
    write_tfrecords("/raid/data/imseg/preproc-data", "/raid/data/imseg/tfrecord-data", crop = False, reshape = False)
    # write_tfrecords("/preproc-data", "/data")

    

