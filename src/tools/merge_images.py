#!/usr/bin/env python

from PIL import Image


#Assume all images passed into the list are the same size
class img_merger():
	def __init__(self, plot_path_list, crop_window=None, horizontal=True):
		self.plot_path_list = plot_path_list
		self.num_of_plots = len(plot_path_list)
		self.crop_window = crop_window
		self.horizontal = horizontal
		self.img_list = []

		self.open_all_images()
		self.crop_img()
		self.size = self.img_list[0].size

		exec('self.merge_' + str(self.num_of_plots) + '()')

	def open_all_images(self):
		for pth in self.plot_path_list:
			im = Image.open(pth)
			self.img_list.append(im)


	def crop_img(self):
		if self.crop_window is None: return
		for idx, img in enumerate(self.img_list):
			self.img_list[idx] = img.crop(self.crop_window)


	def show_img(self):
		self.output_img.show()

	def merge_2(self):
		if self.horizontal:
			new_width = 2*self.size[0]
			new_height = self.size[1]
	
			newImg = Image.new('RGB', (new_width, new_height))
			newImg.paste(self.img_list[0], (0,0))
			newImg.paste(self.img_list[1], (self.size[0],0))
	
			self.output_img = newImg
		else:
			new_width = self.size[0]
			new_height = 2*self.size[1]
	
			vert_img = Image.new('RGB', (new_width, new_height))
			vert_img.paste(self.img_list[0], (0,0))
			vert_img.paste(self.img_list[1], (0,self.size[1]))
	
			self.output_img = vert_img


	def merge_3(self):
		if self.horizontal:
			new_width = 3*self.size[0]
			new_height = self.size[1]
	
			newImg = Image.new('RGB', (new_width, new_height))
			newImg.paste(self.img_list[0], (0,0))
			newImg.paste(self.img_list[1], (self.size[0],0))
			newImg.paste(self.img_list[2], (2*self.size[0],0))
	
			self.output_img = newImg
		else:
			new_width = self.size[0]
			new_height = 3*self.size[1]
	
			vert_img = Image.new('RGB', (new_width, new_height))
			vert_img.paste(self.img_list[0], (0,0))
			vert_img.paste(self.img_list[1], (0,self.size[1]))
			vert_img.paste(self.img_list[2], (0,2*self.size[1]))
	
			self.output_img = vert_img

	def merge_4(self):
		new_width = 2*self.size[0]
		new_height = 2*self.size[1]

		newImg = Image.new('RGB', (new_width, new_height))
		newImg.paste(self.img_list[0], (0,0))
		newImg.paste(self.img_list[1], (self.size[0],0))
		newImg.paste(self.img_list[2], (0, self.size[1]))
		newImg.paste(self.img_list[3], (self.size[0], self.size[1]))

		self.output_img = newImg


	def merge_5(self):
		img_w = self.size[0]
		img_h = self.size[1]

		new_width = 3*img_w
		new_height = 2*img_h

		newImg = Image.new('RGB', (new_width, new_height), 'WHITE')
		newImg.paste(self.img_list[0], (0,0))
		newImg.paste(self.img_list[1], (img_w,0))
		newImg.paste(self.img_list[2], (2*img_w,0))
		newImg.paste(self.img_list[3], (int(img_w/2.0), img_h))
		newImg.paste(self.img_list[4], (int(3*img_w/2.0), img_h))

		self.output_img = newImg

	def merge_6(self):
		img_w = self.size[0]
		img_h = self.size[1]

		if self.horizontal:
			new_width = 3*img_w
			new_height = 2*img_h
	
			newImg = Image.new('RGB', (new_width, new_height), 'WHITE')
			newImg.paste(self.img_list[0], (0,0))
			newImg.paste(self.img_list[1], (img_w,0))
			newImg.paste(self.img_list[2], (2*img_w,0))
			newImg.paste(self.img_list[3], (0, img_h))
			newImg.paste(self.img_list[4], (img_w, img_h))
			newImg.paste(self.img_list[5], (2*img_w, img_h))
	
			self.output_img = newImg
		else:
			new_width = 2*img_w
			new_height = 3*img_h
	
			newImg = Image.new('RGB', (new_width, new_height), 'WHITE')
			newImg.paste(self.img_list[0], (0,0))
			newImg.paste(self.img_list[1], (img_w,0))
			newImg.paste(self.img_list[2], (0,img_h))
			newImg.paste(self.img_list[3], (img_w, img_h))
			newImg.paste(self.img_list[4], (0, 2*img_h))
			newImg.paste(self.img_list[5], (img_w, 2*img_h))
	
			self.output_img = newImg



	def save_img(self, output_path):
		self.output_img.save(output_path)
	

if __name__ == "__main__":

	img_path_list = []
	#img_path_list.append("wine/wine_1/kernel_1.png")
	#img_path_list.append("wine/wine_1/kernel_2.png")
	img_path_list.append("../../results/spiral/spiral_1/score_layer_progression.png")
	img_path_list.append("../../results/wine/wine_1/score_layer_progression.png")
	img_path_list.append("../../results/cancer/cancer_1/score_layer_progression.png")
	img_path_list.append("../../results/car/car_1/score_layer_progression.png")
	img_path_list.append("../../results/divorce/divorce_1/score_layer_progression.png")
	img_path_list.append("../../results/face/face_1/score_layer_progression.png")



	imSize = Image.open(img_path_list[0]).size
	#crop_window = (80, 20,imSize[0] - 80, imSize[1] - 5)
	crop_window = (20, 10,imSize[0] - 20, imSize[1] - 5)

	Imerger = img_merger(img_path_list, crop_window, horizontal=False)
	#Imerger.show_img()
	Imerger.save_img('combined_layer_progression.png')


