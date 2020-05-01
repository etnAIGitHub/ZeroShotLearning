from numpy import genfromtxt
from os import path, listdir
import cv2
import pickle
from tqdm import tqdm
from threading import Thread, Lock
from random import sample
from math import ceil
tqdm_mutex = Lock()
from shutil import copy

class BenedettiniFilter():

    class FilterThread(Thread):
        def __init__(self, target_dir, subset, filter_function, tqdm, copy_filtered=False):
            Thread.__init__(self)
            self.target_dir = target_dir
            self.subset = subset
            self.filter_function = filter_function
            self._return = None
            self.tqdm = tqdm
        
        def run(self):
            unfiltered_subset_list = []
            unfiltered_per_class_count = [0] * 35
            per_class_count = [0] * 35

            for folder_name in self.subset:
                try:
                    images_dir = path.join(self.target_dir, str(folder_name))
                    target_labels = genfromtxt(path.join(self.target_dir, 'labels/%d.txt' % folder_name))
                    target_images = [filename for filename in listdir(images_dir)]
            
                    if len(target_labels) != len(target_images):
                        raise Exception("error labels and images num are not the same!")

                    for index, target_path in enumerate(target_images):
                        target_img_path = path.join(str(folder_name), target_path)
                        target_class = int(target_labels[index])

                        per_class_count[target_class] += 1

                        if not self.filter_function(path.join(self.target_dir, target_img_path)):
                            unfiltered_subset_list.append((target_img_path, target_class))
                            # count how many images per class pass the filter
                            unfiltered_per_class_count[target_class] += 1
                        else if copy_filtered:
                            destination = r"E:/Sources/Repos/filtered-benedettini/filtered/"+str(folder_name)+'_'+path.basename(target_img_path)
                            copy(path.join(self.target_dir, target_img_path), destination)

                    tqdm_mutex.acquire()
                    self.tqdm.update(1)
                    tqdm_mutex.release()

                except Exception as ex:
                    print(ex)

            self._return = (unfiltered_subset_list, unfiltered_per_class_count, per_class_count)

        def join(self, *args):
            Thread.join(self, *args)
            return self._return

    def __init__(self, root_dir, num_thread=8):
        self.target_dir = path.join(root_dir, 'Test')
        self.num_thread = num_thread

        folders_name_range = set(range(100, 166+1))
        #since some folders are missing, remove them from set
        missing_folders_name = {106, 127, 128, 131, 133, 141, 163}
        self.folders_set = folders_name_range - missing_folders_name

        self.run()
        self.join()

    def run(self):
        # Return a list of num_thread folder_subset
        pbar = tqdm(total=len(self.folders_set))
        self.folder_subset = self.computeThreadSubset(self.num_thread)
        
        # Create and run threads
        self.filter_threads = [None] * self.num_thread
        for i in range(self.num_thread):
            self.filter_threads[i] = self.FilterThread(self.target_dir, self.folder_subset[i], self.check_filter, tqdm=pbar)
            self.filter_threads[i].start()

    def saveResults(self, unfiltered_list, unfiltered_per_class_counter, per_class_counter):
        pickle_out = open("unfiltered_list.pickle","wb")
        pickle.dump(unfiltered_list, pickle_out)
        pickle_out.close()

        pickle_out = open("unfiltered_per_class_counter.pickle","wb")
        pickle.dump(unfiltered_per_class_counter, pickle_out)
        pickle_out.close()

        pickle_out = open("per_class_counter.pickle","wb")
        pickle.dump(per_class_counter, pickle_out)
        pickle_out.close()

    def join(self):
        unfiltered_list = []
        unfiltered_per_class_counter = [0] * 35
        per_class_counter = [0] * 35

        for i in range(self.num_thread):
            unfiltered_subset_list, unfiltered_per_class_count, per_class_count = self.filter_threads[i].join()
            unfiltered_list += unfiltered_subset_list
            unfiltered_per_class_counter = [x + y for x, y in zip(unfiltered_per_class_counter, unfiltered_per_class_count)]
            per_class_counter = [x + y for x, y in zip(per_class_counter, per_class_count)]
            
        print('unfiltered_per_class_counter', unfiltered_per_class_counter)
        print('per_class_counter', per_class_counter)

        self.saveResults(unfiltered_list, unfiltered_per_class_counter, per_class_counter)

    def check_filter(self, imagePath):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, std = cv2.meanStdDev(gray)
        height, width, _ = image.shape
        pixel_count = height*width
        max_shape = max(height, width)
        min_shape = min(height, width)
        rateo = max_shape/float(min_shape)
        laplacian_var  = cv2.Laplacian(image, cv2.CV_64F).var()

        if pixel_count < 86*86 or rateo > 4 or std[0][0] < 6 or laplacian_var < 10:
            return True

        return False

    def computeThreadSubset(self, num_thread):
        '''
        # Arguments
            num_thread: number of thread to use
        # Returns
            folder_subset: list of size (num_thread) containing folders name that each thread should work with
        '''
        # Make a copy to avoid distruction of the original set
        folders_set_tmp = self.folders_set.copy()

        #split set into num_thread subset of almost equals size
        folder_subset_len = ceil(len(folders_set_tmp) / num_thread)
        folder_subset = [None] * num_thread
        for i in range(num_thread):
            if i < num_thread-1:
                folder_subset[i] = set(sample(folders_set_tmp, folder_subset_len))
                folders_set_tmp -= folder_subset[i]
            else: #last set contains remaining folders
                folder_subset[i] = folders_set_tmp

        return folder_subset

if __name__ == "__main__":
    b_filter = BenedettiniFilter(
        root_dir = r"E:\Sources\Repos\filtered-benedettini\Object Retrieval\Monastero dei Benedettini",
        num_thread = 8,
    )