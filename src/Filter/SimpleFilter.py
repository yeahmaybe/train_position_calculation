class SimpleFilter:
    __k = 0.65

    def filtrate(self, angle, previous):
        k = self.__k
        if previous is not None:
            return k * angle + (1 - k) * previous
        return angle
