

import queue

complaints = queue.PriorityQueue()

complaints.put((30, "Price is too high"))
complaints.put((10, "Service is terrible!  I'm definitely not renewing my account!"))
complaints.put((20, "Not sure if I want to renew my account"))

