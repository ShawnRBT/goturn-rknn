# Date: Friday 02 June 2017 05:50:20 PM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Test file for showing the tracker output


# import argparse
# import setproctitle
from logger.logger import setup_logger
from network.regressor_try import regressor
from loader.loader_vot import loader_vot
from loader.loader_visdrone import loader_visdrone
from tracker.tracker import tracker
from tracker.tracker_manager import tracker_manager
from rknn.api import RKNN

# setproctitle.setproctitle('TEST_VIDEO')
logger = setup_logger(logfile=None)

prototxt = '/home/toybrick/Desktop/mono-track/rknn-model/tracker.prototxt'
model =  '/home/toybrick/Desktop/mono-track/rknn-model/goturn.rknn'
gpuID = 0
# video_folder = '/home/toybrick/Desktop/mono-track/video/sequences'
# annotation_folder = '/home/toybrick/Desktop/mono-track/video/annotations'
video_folder = '/home/toybrick/Desktop/video/sequences'
annotation_folder = '/home/toybrick/Desktop/video/annotations'
save_path = '/home/toybrick/Desktop/mono-track/result'



do_train = False
objRegressor = regressor(prototxt, model, gpuID, 1, do_train, logger)
objTracker = tracker(False, logger)  # Currently no idea why this class is needed, eventually we shall figure it out
objLoaderVisdrone = loader_visdrone(video_folder, annotation_folder, logger)
objLoaderVisdrone.loaderVisdrone()
videos = objLoaderVisdrone.get_videos()
objTrackerVis = tracker_manager(videos, objRegressor, objTracker, logger, save_path)
objTrackerVis.trackAll(0, 1)
