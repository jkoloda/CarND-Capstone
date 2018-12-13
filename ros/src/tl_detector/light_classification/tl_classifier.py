from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy
import cv2
import os


MAX_IMAGE_WIDTH = 300
MAX_IMAGE_HEIGHT = 300


class TLClassifier(object):
    """Traffic light classifier based on a tensorflow model."""

    def __init__(self, is_site=True):
        """Build, load and prepare traffic light classifier object.

        Loads classifier trained on simulator or real data, depending on the
        is_site flag coming from the configuration file.

        """
        self.session = None
        self.detection_graph = None

        self.classes = {1: TrafficLight.RED,
                        2: TrafficLight.YELLOW,
                        3: TrafficLight.GREEN,
                        4: TrafficLight.UNKNOWN}

        self.light_labels = ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']

        temp = os.path.dirname(os.path.realpath(__file__))
        temp = temp.replace(
                'ros/src/tl_detector/light_classification',
                'models',
                )

        if is_site is False:
            self.model_path = os.path.join(temp,
                                           'frozen_inference_graph_sim.pb')
        else:
            self.model_path = os.path.join(temp,
                                           'frozen_inference_graph_real.pb')

        self.load_model(model_path=self.model_path)

    def get_classification(self, image):
        """Determine the color of the traffic light in the image.

        Args
        ----
            image (cv::Mat): image containing the traffic light

        Returns
        -------
            int: ID of traffic light color
                 (specified in styx_msgs/TrafficLight)

        """
        class_idx, confidence = self.predict(image)
        return class_idx

    def load_model(self, model_path):
        """Load classifier (graph and session)."""
        self.detection_graph = tf.Graph()
        with tf.Session(graph=self.detection_graph) as sess:
            self.session = sess
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def predict(self, image_np, min_score_thresh=0.5):
        """Predict traffic light state from image.

        Parameters
        ----------
        image_np : ndarray
            Input image.

        min_score_threshold : float
            Confidence threshold for traffic light classification.

        Returns
        -------
        light : TrafficLight
            Light color of traffic light detected on input image.

        score : float
            Classification confidence score.

        """
        image_tensor = self.detection_graph.\
            get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.\
            get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.\
            get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.\
            get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.\
            get_tensor_by_name('num_detections:0')
        image_np = self.process_image(image_np)

        input = [detection_boxes, detection_scores, detection_classes]
        (boxes, scores, classes) = self.session.run(
                input,
                feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)

        # Traffic light state decision
        # In case mutliple traffic lights are detected (as e.g. is the case of
        # the simulator) we select the light with the highest accumulated score
        accumulated_scores = np.zeros(len(self.classes))
        accumulated_classes = np.zeros(len(self.classes))
        for ii, score in enumerate(scores):
            if score > min_score_thresh:
                # light_class = self.classes[classes[ii]]
                # return light_class, score
                rospy.loginfo(self.light_labels[int(classes[ii] - 1)])

                accumulated_scores[classes[ii] - 1] += score
                accumulated_classes[classes[ii] - 1] += 1

        if np.sum(accumulated_scores) > 0:
            light_class_idx = np.argmax(accumulated_scores) + 1
            confidence = accumulated_scores[light_class_idx - 1] / \
                float(accumulated_classes[light_class_idx - 1])

            return self.classes[light_class_idx], confidence

        else:
            return None, None

    def process_image(self, img):
        """Pre-process imae so it can be passed directly to classifier.

        Pre-processing consists of shrinkng the image to default maximum size
        and converting in to RGB format (assuming that input is BGR).

        Parameters
        ----------
        img : ndarray
            Input image to be processed.

        Returns
        -------
        img : ndarray
            Processed image.

        """
        img = cv2.resize(img, (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def shrink_image(self, img):
        """Shrink image if bigger than default maximum dimensions.

        Aspect ratio is kept. If the image is smaller it is return as it is.

        Parameters
        ----------
        img : ndarray
            Input image to be shrinked if necessary.

        Returns
        -------
        img : ndarray
            Shrinked image.

        """
        height, width = img.shape[:2]
        if MAX_IMAGE_HEIGHT < height or MAX_IMAGE_WIDTH < width:
            scaling_factor = np.min(MAX_IMAGE_HEIGHT / float(height),
                                    MAX_IMAGE_WIDTH / float(width))
            img = cv2.resize(img, None, fx=scaling_factor,
                             fy=scaling_factor, interpolation=cv2.INTER_AREA)

        return img
