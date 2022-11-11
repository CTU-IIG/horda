class Result:
    def __init__(self, criterion, time, order):
        self.order = order
        self.time = time
        self.criterion = criterion

    def to_mongo(self):
        return self.__dict__

    def empty(self):
        return self.order is None and self.time is None and self.criterion is None

    def __str__(self):
        return f'(c:{self.criterion}, t:{self.time}, o:{self.order})'

    def to_complete_result(self, name):
        return CompleteResult(self.criterion, self.time, self.order, estimator_name=name)

    def to_human(self):
        return {'criterion': self.criterion, 'order': self.order, 'time': self.time}


class CompleteResult(Result):
    def __init__(self, criterion, time, order, comment='', bonus=None, estimator_name=None, **kwargs):
        super().__init__(criterion, time, order)
        self.estimator_name = estimator_name
        self.comment = comment
        self.bonus = bonus
        self.one = 1

    def __str__(self):
        return f'(c:{self.criterion}, t:{self.time}, o:{self.order})'

    def __repr__(self):
        return str(self)

    def to_complete_result(self, name):
        return CompleteResult(self.criterion, self.time, self.order, comment=self.comment, bonus=self.bonus,
                              estimator_name=name)


ZERO_RESULT = Result(0, 0, tuple())
