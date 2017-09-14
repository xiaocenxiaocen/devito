class Pickleable(object):
    __pickled = []
    def __getstate__(self):
        return [getattr(self, item) for item in self._pickled]

    def __setstate__(self, state):
        self.__init__(*state)
