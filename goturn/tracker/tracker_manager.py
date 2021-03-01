# Date: Wednesday 07 June 2017 11:28:11 AM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: tracker manager

import cv2
import os
import numpy as np
from rknn.api import RKNN

opencv_version = cv2.__version__.split('.')[0]


class tracker_manager:

    """Docstring for tracker_manager. """

    def __init__(self, videos, regressor, tracker, logger, save_path):
        """This is

        :videos: list of video frames and annotations
        :regressor: regressor object
        :tracker: tracker object
        :logger: logger object
        :returns: list of video sub directories
        """

        self.videos = videos
        self.regressor = regressor
        self.tracker = tracker
        self.logger = logger
        self.save_path = save_path
        print("save_path", save_path)
        

    def trackAll(self, start_video_num, pause_val):
        """Track the objects in the video
        """

        videos = self.videos
        objRegressor = self.regressor
        objTracker = self.tracker

        count = 0

        rknn = RKNN(verbose=True, verbose_file='mono-track/rknn.log')
        # load rknn model
        print('--> Loading rknn model')
        rknn.load_rknn(objRegressor.rknn_model)
        print('done')

        # init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        for i in range(start_video_num, len(videos)):
            video_frames = videos[i].all_frames
            annot_frames = videos[i].annotations

            num_frames = min(len(video_frames), len(annot_frames))

            first_frame = cv2.imread(videos[i].all_frames[0])
            print("frame shape:", first_frame.shape)
            self.fream_size = (first_frame.shape[1], first_frame.shape[0])
            vw = cv2.VideoWriter(os.path.join(self.save_path, 'result{}.avi'.format(count)), cv2.VideoWriter_fourcc(*'MJPG'), 30.0, self.fream_size, True)
            # Get the first frame of this video with the intial ground-truth bounding box
            frame_0 = video_frames[0]
            bbox_0 = annot_frames[0].bbox
            sMatImage = cv2.imread(frame_0)
            objTracker.init(sMatImage, bbox_0, objRegressor)
            for j in range(1, num_frames):
                frame = video_frames[j]
                sMatImage = cv2.imread(frame)
                sMatImageDraw = sMatImage.copy()
                bbox = annot_frames[j].bbox
                

                if opencv_version == '2':
                    cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 255, 255), 2)
                else:
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 255, 255), 2)

                bbox = objTracker.track(sMatImage, objRegressor, rknn)
                print("track_bbox:[{}, {}, {}, {}]".format(bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                print("frame shape", sMatImageDraw.shape)
                if opencv_version == '2':
                    cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)
                else:
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)

                if not self.save_path:
                    cv2.imshow('Results', sMatImageDraw)
                    cv2.waitKey(10)
                else:
                    vw.write(sMatImageDraw)

            vw.release()
            print("saving result{}.avi in {}".format(count, self.save_path))
            count += 1
