from HandGestureRecognition import GestureRecognition


class Fingers_5_2_exercise(GestureRecognition):
    def __init__(self) -> None:
        super().__init__()

    def check_5_plus_2_performance(self, lmkArr):
        finger_status_list = self._finger_status(lmkArr)
        thumbOpen, firstOpen, secondOpen, thirdOpen, fourthOpen = finger_status_list
        number = sum(finger_status_list)
        if number ==5:
            return 'Five', 5
        elif number ==2:
            if firstOpen and secondOpen:
                return 'Two',2
            else:
                return 'Raise index and middle fingers to make number two', None
        else:
            return f'{number}, Raise 5 or 2 fingers (index and middle)', None

