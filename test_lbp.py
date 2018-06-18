import os
import argparse
import numpy as np
import cv2 as cv

class LBP:
	def __init__(self, image, output_name, p, r):
		img = cv.imread(image)
		self.input_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		self.output_image = self.input_image.copy()
		self.output_name = output_name

		self.num_pix = int(p)
		self.radius = int(r)


	def choose_pixels(self, w, h):
		pixel_list = []

		for i in range(self.num_pix):
			pixel = (int(w + self.radius * np.cos(2 * np.pi * i / self.num_pix)),
					int(h + self.radius * np.sin(2 * np.pi * i / self.num_pix)))
			pixel_list.append(pixel)

		return pixel_list


	def lbp(self, pixel_list, thresh):
		lbp_val = 0
		for i in range(len(pixel_list)):
			pixel_val = self.input_image[pixel_list[i][1], pixel_list[i][0]]
			lbp_val += (pixel_val > thresh) * (2 ** i)

		return lbp_val


	def create_lbp_image(self):
		(iH, iW) = self.input_image.shape
		num_lbps = 0
		for h in range(self.radius + 1, iH - self.radius - 1):
			for w in range(self.radius + 1, iW - self.radius - 1):
				pixel_list = self.choose_pixels(w, h)
				thresh = self.input_image[h, w]
				lbp_val = self.lbp(pixel_list, thresh)

				self.output_image[h, w] = lbp_val
				num_lbps += 1
				print("lbp: %d, %d / %d" % (lbp_val, num_lbps, (iH - 2 *self.radius - 2) * (iW - 2 * self.radius -2 )))


		cv.imwrite(self.output_name, self.output_image)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=True)
	parser.add_argument("--output", required=True)
	parser.add_argument("--p", default=8)
	parser.add_argument("--r", default=8)

	args = parser.parse_args()
	lbp = LBP(args.input, args.output, args.p, args.r)
	lbp.create_lbp_image()

if __name__ == "__main__":
	main()
	

		
