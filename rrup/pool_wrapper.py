from multiprocessing import Pool


class PoolWrapper:
    def __init__(self, number_of_process=1, debug_mode=False):
        self.np = number_of_process
        self.debug_mode = debug_mode
        if self.debug_mode or self.np == 1:
            self.pool = None
        else:
            self.pool = Pool(self.np)

    def map(self, function, iterable_list):
        result = []
        if self.pool is None:
            print "Ignoring number of processes here, executing for loop"
            for item in iterable_list:
                result.append(function(item))
        else:
            result = self.pool.map(function, iterable_list)

        return result







