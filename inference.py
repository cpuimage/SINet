# -*- coding: utf-8 -*-
import time
import os
import numpy as np
import tensorflow as tf
import cv2


def export_tflite(output_resolution, num_classes, checkpoint_dir):
    from model import Model
    model = Model(output_resolution=output_resolution, num_classes=num_classes)
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        tf.get_logger().info("Latest checkpoint restored:{}".format(ckpt_manager.latest_checkpoint))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        tflite_model = converter.convert()
        open("model.tflite", "wb").write(tflite_model)
        tf.get_logger().info('export tflite done.')

    else:
        tf.get_logger().info('Not restoring from saved checkpoint')


class Segmentator(object):
    def __init__(self):
        self.height = 256
        self.width = 256

    def load_tflite(self, model_path="model.tflite"):
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get Input shape
        self.input_shape = self.input_details[0]['shape']
        self.height, self.width = self.input_shape[1], self.input_shape[2]

    def tflite_forward(self, img_data):
        input_data = img_data.reshape(self.input_shape).astype('float32')

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        mask = (self.interpreter.get_tensor(self.output_details[0]['index']))

        return mask

    def cv_load_image_rgb(self, filename):
        image = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return np.asarray(image, np.float32)

    def build_debug_display(self, img_data, disp_image, disp_mask, h, w, maintain_resolution=True):
        if maintain_resolution:
            disp_mask = np.clip(np.expand_dims(cv2.resize(disp_mask, (w, h), interpolation=cv2.INTER_CUBIC), -1), 0,
                                1)
            disp_image = cv2.cvtColor(np.asarray(img_data, np.uint8), cv2.COLOR_RGBA2RGB)
        else:
            disp_image = cv2.cvtColor(np.asarray(disp_image * 255., np.uint8), cv2.COLOR_RGBA2RGB)
        overlay = disp_image.copy()
        disp_mask_rgb = (np.tile(disp_mask, 3) * 255.).astype(np.uint8)
        disp_image_mask = (disp_image * disp_mask).astype(np.uint8)
        overlay = (np.broadcast_to([1, 0, 1], overlay.shape) * disp_image_mask).astype(np.uint8)
        alpha = 0.7
        cv2.addWeighted(disp_image, alpha, overlay, 1 - alpha, 0, overlay)
        extracted_pixels_color = np.broadcast_to([[207, 207, 207]], overlay.shape) * (1. - disp_mask)
        extracted_pixels = extracted_pixels_color + disp_image_mask
        outputs = np.uint8(disp_image), np.uint8(disp_mask_rgb), np.uint8(
            overlay), np.uint8(extracted_pixels)
        return outputs

    def run_inference(self, data, only_mask=True, debug_display=True):
        if isinstance(data, str):
            img_data = tf.cast(self.cv_load_image_rgb(data), tf.float32)
        else:
            img_data = data
        height, width = img_data.shape[:2]
        disp_image = np.asarray(tf.image.resize(img_data / 255., size=[int(self.height), int(self.width)]), np.float32)
        start = time.perf_counter()
        mask = self.tflite_forward(disp_image)
        print('Time:        {:.3f} secs'.format(time.perf_counter() - start))
        disp_mask = np.squeeze(np.clip(mask, 0., 1.), 0)
        if debug_display:
            alpha_disp_image, alpha_disp_mask_rgb, alpha_overlay, alpha_extracted_pixels = self.build_debug_display(
                img_data, disp_image, disp_mask, height, width)
            outputs = np.concatenate((alpha_disp_image, alpha_disp_mask_rgb, alpha_overlay, alpha_extracted_pixels),
                                     axis=1)
            return outputs.astype(np.uint8)
        mask = np.asarray(tf.image.resize(disp_mask, size=[height, width]) * 255., np.uint8)
        if only_mask:
            outputs = mask
        else:
            outputs = np.concatenate((img_data, mask), axis=-1)
        return outputs.astype(np.uint8)


def main():
    output_resolution = 512
    num_classes = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    export_model = True
    if export_model:
        checkpoint_path = "training/cp-{step:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        export_tflite(output_resolution, num_classes, checkpoint_dir=checkpoint_dir)
    print('************ Segnet ************')

    segnet = Segmentator()
    segnet.load_tflite()
    root_path = "./data/test/coco_test"
    image_path = os.path.join(root_path, "image")
    save_path = os.path.join(root_path, "probs")
    import shutil
    shutil.rmtree(save_path, True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sub_file = os.listdir(image_path)
    for name in sub_file:
        print("Working on file", name)
        input_filename = os.path.join(image_path, name)
        png_filename = os.path.splitext(name)[0] + ".png"
        save_filename = os.path.join(save_path, png_filename)
        if os.path.exists(input_filename) and not os.path.exists(save_filename):
            probs = segnet.run_inference(input_filename, only_mask=True)
            png_image = tf.image.encode_png(probs)
            tf.io.write_file(save_filename, png_image)


if __name__ == '__main__':
    main()
