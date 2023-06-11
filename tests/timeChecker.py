import time

class Timer:
	startTime = 0
	def __init__(self) -> None:
		self.startTime = time.time()

	def reset(self):
		self.startTime = time.time()
		return True
	
	def checkNreset(self):
		print(f"\r\rTime : {time.time() - self.startTime} sec")
		self.reset()