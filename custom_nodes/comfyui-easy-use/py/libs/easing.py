@staticmethod
def easyIn(t: float)-> float:
    return t*t
@staticmethod
def easyOut(t: float)-> float:
    return -(t * (t - 2))
@staticmethod
def easyInOut(t: float)-> float:
    if t < 0.5:
        return 2*t*t
    else:
        return (-2*t*t) + (4*t) - 1

class EasingBase:

    def easing(self, t: float, function='linear') -> float:
        if function == 'easyIn':
            return easyIn(t)
        elif function == 'easyOut':
            return easyOut(t)
        elif function == 'easyInOut':
            return easyInOut(t)
        else:
            return t

    def ease(self, start, end, t) -> float:
        return end * t + start * (1 - t)