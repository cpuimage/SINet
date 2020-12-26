import cv2, time
import numpy as np
import os
import tensorflow as tf


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

    def run_inference(self, data, only_mask=True, debug_display=False):
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
    cam_width = 320 * 2
    cam_height = 240 * 2
    fps = ""
    cap = cv2.VideoCapture(0)
    background_image = cv2.imread('background.jpg')
    apply_background = True
    if background_image is not None:
        background_image = tf.image.resize(tf.cast(background_image, tf.float32), (cam_height, cam_width)).numpy()
    else:
        apply_background = False
    overlay = np.zeros((cam_height, cam_width, 3), np.uint8)
    overlay[:] = (1, 0, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cv2.namedWindow('FPS', cv2.WINDOW_AUTOSIZE)
    segnet = Segmentator()
    segnet.load_tflite()

    inv_255 = 1. / 255.
    while True:
        start = time.time()
        ret, frame = cap.read()
        img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = segnet.run_inference(data=img_data, only_mask=True)
        if apply_background:
            alpha = cv2.cvtColor(outputs, cv2.COLOR_GRAY2BGR) * inv_255
            output = (background_image * (1. - alpha) + frame * alpha).astype(np.uint8)
        else:
            mask = cv2.cvtColor(outputs, cv2.COLOR_GRAY2BGR) * overlay
            output = cv2.addWeighted(frame, 1, mask, 0.9, 0)
        cv2.putText(output, fps, (cam_width - 180, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('FPS', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elapsed_time = time.time() - start
        fps = "(Playback) {:.1f} FPS".format(1 / elapsed_time)
        print("fps = ", str(fps))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()
