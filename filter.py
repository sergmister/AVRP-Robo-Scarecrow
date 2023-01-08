import pyzed
import pyzed.sl as sl
from time import perf_counter
from functools import cmp_to_key

class Filter():

    def __init__(self):
        self.max_time = 3
        self.queue = []
        self.track_start_time = perf_counter()
        self.current_target = None

    def cmp(self, item1, item2):
        return item1.confidence - item2.confidence

    def getTarget(self, objects):
        temp = sl.ObjectData()
        if len(objects.object_list) > 0:
            if self.current_target is not None:
                if objects.get_object_data_from_id(temp, self.current_target) and (perf_counter() - self.track_start_time) < self.max_time and temp.confidence > 0.3:
                    # target is the same; return the same target, and None to indicate the track_start_time should not be updated
                    return temp

            # finding new target
            # sort the object list by confidence
            # enqueue everything not already in queue
            obj_list = sorted(objects.object_list, key=cmp_to_key(self.cmp))
            for object in obj_list:
                if object not in self.queue and object.confidence > 0.3:
                    self.queue.append(object)


            # pop until we find an object that is still being tracked
            target = None
            while target == None:
                top = self.queue.pop(0)
                if objects.get_object_data_from_id(temp, top.id):
                    target = top

            # new target
            self.track_start_time = perf_counter()
            self.current_target = target.id
            return target

        # no object
        else:
            return None