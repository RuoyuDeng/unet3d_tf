import os
import numpy as np
import tensorflow as tf
import random
import argparse
from IPython import embed

def randrange(max_range):
    return 0 if max_range == 0 else random.randrange(max_range)


def get_cords(cord, idx):
    patch_size = [186, 186, 128]
    return cord[idx], cord[idx] + patch_size[idx]

def min_shape(np_files_path):
    files = os.listdir(np_files_path)
    files.sort()

    imgs = [os.path.join(np_files_path, file) for file in files if "_x" in file]  # list of file names (strings)


    img_shapes = [np.load(img).shape for img in imgs]
    shapes_np = np.array(img_shapes)
    print(np.amin(shapes_np,axis=0))

def npy_to_tfrecords(prep_np_dir, output_dir):
    # list all files
    files = os.listdir(prep_np_dir)
    files.sort()

    print(files)

    store_imgs = [os.path.join(prep_np_dir, file) for file in files if "_x" in file]
    store_lbls = [os.path.join(prep_np_dir, file) for file in files if "_y" in file]

    img_list = []
    lbl_list = []
    mean_list = []
    std_list = []

    for i, img in enumerate(store_imgs):
        print("Starting case", i + 1)
        # image_array = np.load(img)  # array type
        
        image_array = np.load(img).astype(np.float32)
        label_array = np.load(store_lbls[i]).astype(np.uint8)  # array type
        print(image_array.shape,label_array.shape)
        mean = np.mean(image_array)
        std = np.std(image_array)
        
        
        img_list.append(image_array)
        lbl_list.append(label_array)
        mean_list.append(mean)
        std_list.append(std)
        # embed()
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
            # change from tostring() to tobytes(), suggested by numpy
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

def write_tfrecords(np_files_path, output_dir, crop = False, reshape = False):
    """ Convert numpy array to rfrecord

    :param np_files_path: String representing path where np files are stored
    :param output_dir: String representing path where to write
    """
    # create directory to store reshaped np-files
    prep_np_dir = os.path.join(np_files_path,f"{raw_np_dir}-prep")
    if not os.path.exists(prep_np_dir):
        os.makedirs(prep_np_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # list all files
    files = os.listdir(np_files_path)
    files.sort()

    # split files into samples list and labels list
    imgs = [os.path.join(np_files_path, file) for file in files if "_x" in file]  # list of file names (strings)
    lbls = [os.path.join(np_files_path, file) for file in files if "_y" in file]  # list of file names (strings)
    
    store_imgs = [os.path.join(prep_np_dir, file) for file in files if "_x" in file]
    store_lbls = [os.path.join(prep_np_dir, file) for file in files if "_y" in file]

    if reshape:
        print('Starting reshaping')
        for i in range(len(imgs)):
            print(f'Reshaping case {i}')
            cur_img = np.load(imgs[i])
            cur_img = np.moveaxis(cur_img, 0, -1) # 1,190,392,392 -> 190,392,392,1
            new_img = np.moveaxis(cur_img, 0, 2) # 392,392,190
            new_lbl = np.squeeze(np.load(lbls[i]), 0) # 1,190,392,392 -> 190,392,392
            new_lbl = np.moveaxis(new_lbl, 0, 2) # 190,392,392 -> 392,392,190
            np.save(store_imgs[i], new_img)
            np.save(store_lbls[i], new_lbl)

    if crop:
        print('Starting croping')
        for i in range(len(store_imgs)):
            print(f'Cropping case {i}')
            img = np.load(store_imgs[i])
            lbl = np.load(store_lbls[i])
            ranges = [s - p for s, p in zip(img.shape[:-1], [186, 186, 128])]
            cord = [randrange(x) for x in ranges]
            low_x, high_x = get_cords(cord, 0)
            low_y, high_y = get_cords(cord, 1)
            low_z, high_z = get_cords(cord, 2)
            new_img = img[low_x:high_x, low_y:high_y, low_z:high_z, :]
            new_lbl = lbl[low_x:high_x, low_y:high_y, low_z:high_z]
            np.save(store_imgs[i], new_img)
            np.save(store_lbls[i], new_lbl)

    npy_to_tfrecords(prep_np_dir,output_dir)
    


if __name__ == "__main__":
    params = argparse.ArgumentParser(description="Prepare tf-data for unet3d_tf version")
    params.add_argument("--np-raw", type=str)
    params.add_argument("--tf-data",type=str)
    params.add_argument("--do-crop",action="store_true", default=False)
    params.add_argument("--do-reshape",action="store_true", default=False)

    args = params.parse_args()
    raw_np_dir = args.np_raw
    tf_dir = args.tf_data
    do_crop = args.do_crop
    do_reshape = args.do_reshape

    # min_img_shape,min_lbl_shape = min_shape(raw_np_dir)
    # max_img_shape,max_lbl_shape = max_shape(raw_np_dir)

    # min_shape(raw_np_dir)
    # print(f"The max image shape: {max_img_shape} ,max label shape: {max_lbl_shape} \n min image shape: {min_img_shape}, min label shape: {min_lbl_shape}")
    write_tfrecords(raw_np_dir,tf_dir, do_crop, do_reshape)
    # npy_to_tfrecords("/npy-prep","/data")

    

