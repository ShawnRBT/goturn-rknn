# Date: Friday 02 June 2017 05:50:20 PM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Test file for showing the tracker output

import argparse
import setproctitle
from ..logger.logger import setup_logger
from ..network.regressor import regressor
from ..loader.loader_vot import loader_vot
from ..loader.loader_visdrone import loader_visdrone
from ..tracker.tracker import tracker
from ..tracker.tracker_manager import tracker_manager

setproctitle.setproctitle('TEST_VIDEO')
logger = setup_logger(logfile=None)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="Path to the prototxt")
ap.add_argument("-m", "--model", required=True, help="Path to the model")
ap.add_argument("-v", "--video_folder", required=True, help="Path to the test video folder")
ap.add_argument("-a", "--annotation_folder", required=True, help="path to the test annotations folder")
ap.add_argument("-g", "--gpuID", required=True, help="gpu to use")
ap.add_argument("-s", "--save_path", required=True, help="result saved path")
args = vars(ap.parse_args())

do_train = False
objRegressor = regressor(args['prototxt'], args['model'], args['gpuID'], 1, do_train, logger)
objTracker = tracker(False, logger)  # Currently no idea why this class is needed, eventually we shall figure it out
objLoaderVisdrone = loader_visdrone(args['video_folder'], args['annotation_folder'], logger)
objLoaderVisdrone.loaderVisdrone()
videos = objLoaderVisdrone.get_videos()
objTrackerVis = tracker_manager(videos, objRegressor, objTracker, logger, args['save_path'])
objTrackerVis.trackAll(0, 1)
