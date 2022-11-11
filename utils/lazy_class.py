class LazyClass:
    def force(self):
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')


class ForceClass(LazyClass):

    def __init__(self, method, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def force(self):
        return self.method(*self.args, **self.kwargs)
