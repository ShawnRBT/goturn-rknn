# Date: Nrupatunga: Wednesday 05 July 2017
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: loading Alov dataset

import os
import glob
from loader.video import video
from loader.video import frame
from logger.logger import setup_logger
from helper.BoundingBox import BoundingBox


class loader_visdrone:

    """Docstring for loader_visdrone. """

    def __init__(self, video_folder, annotations_folder, logger):
        """TODO: to be defined1. """

        self.logger = logger
        self.video_folder = video_folder
        self.annotations_folder = annotations_folder
        self.videos = []
        if not os.path.isdir(video_folder):
            logger.error('{} is not a valid directory'.format(video_folder))

    def loaderVisdrone(self):
        """TODO: Docstring for loaderAlov.
        :returns: TODO

        """
        logger = self.logger

        annotations_files = sorted(glob.glob(os.path.join(self.annotations_folder, '*.txt')))
        logger.info('annotations files nums:{0}'.format(len(annotations_files)))

        for ann in annotations_files:
            self.load_annotation_file(ann)


    def load_annotation_file(self, annotation_file):

        video_path = os.path.join(self.video_folder, annotation_file.split('/')[-1].split('.')[0])

        objVideo = video(video_path)
        all_frames = glob.glob(os.path.join(video_path, '*.jpg'))
        objVideo.all_frames = sorted(all_frames)

        with open(annotation_file, 'r') as f:
            data = f.read().rstrip().split('\n')
            frame_num = 0
            for bb in data:
                x, y, w, h = bb.split(',')
                x, y, w, h = int(x), int(y), int(w), int(h)

                x1, y1 = x, y
                x2, y2 = x+w, y+h

                bbox = BoundingBox(x1, y1, x2, y2)
                objFrame = frame(frame_num, bbox)
                objVideo.annotations.append(objFrame)
                frame_num += 1

        video_name = video_path.split('/')[-1]
        self.videos.append(objVideo)

    def get_videos(self, isTrain=True, val_ratio=0.2):
        """TODO: Docstring for get_videos.
        :returns: TODO
        """
        print("start load videos!")
        videos = []
        logger = self.logger

        num_videos = len(self.videos)
        num_val = int(val_ratio * num_videos)
        num_train = num_videos - num_val
        print("num_videos:{}, num_val:{}, num_train:{}".format(num_videos, num_val, num_train))

        if isTrain:
            start_num = 0
            end_num = num_train - 1
        else:
            start_num = num_train
            end_num = num_videos - 1

        for i in range(start_num, end_num + 1):
            video = self.videos[i]
            videos.append(video)
        
        print("videos num:{}".format(end_num - start_num + 1))

        num_annotations = 0
        for i, _ in enumerate(videos):
            num_annotations = num_annotations + len(videos[i].annotations)

        logger.info('Total annotated video frames: {}'.format(num_annotations))

        return videos


if '__main__' == __name__:
    logger = setup_logger(logfile=None)
    objLoaderVisdrone = loader_visdrone('/home/toybrick/Desktop/mono-track/video/sequences', '/home/toybrick/Desktop/mono-track/video/annotations', logger)
    objLoaderVisdrone.loaderVisdrone()
    objLoaderVisdrone.get_videos()
