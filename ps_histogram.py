import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram(img):
	d = {}
	p = []
	upper = np.max(img) + 1
	for i in range(upper):
		d[i] = 0
	for ele in np.nditer(img):
		d[int(ele)] += 1
	total = sum(d.values())
	for i in range(len(d)):
		p.append(d[i]/total)
	return d, p


def equal(filename, s_flag=False):
	s = []
	add = 0
	try:
		image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		bmp = open(filename, 'rb')
		L = 2 ** bmp.read(29)[-1]
		bmp.close()
		m, n = np.shape(image)
		size = m * n
		d, _ = histogram(image)
		for key in d.keys():
			add += (L-1) * d[key] / size
			s.append(round(add))
		if s_flag: return s, L
		new = image.copy()
		for i in range(m):
			for j in range(n):
				new[i, j] = s[new[i, j]]
		_d, _ = histogram(new)
		return d, image, _d, new
	except IOError:
		print('no such file')


def histogram_match(filename, pz):
	assert isinstance(pz, list), 'unexpected data type'
	try:
		img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		m, n = np.shape(img)
		add = 0
		gz = []
		s, l = equal(filename, s_flag=True)
		size = sum(pz)
		for ele in pz:
			add += (l - 1) * ele / size
			gz.append(round(add))
		j = 0
		for i in range(len(s)):
			while j < len(gz):
				if s[i] == gz[j] or s[i] < gz[j+1]:
					s[i] = j
					break
				j += 1
		new = img.copy()
		for i in range(m):
			for j in range(n):
				new[i,j] = s[img[i, j]]
		return img, new
	except IOError:
		print('no such file')


def local_cal(image, L):
	s = []
	add = 0
	m, n = np.shape(image)
	size = m * n
	d, _ = histogram(image)
	for key in d.keys():
		add += (L-1) * d[key] / size
		s.append(round(add))
	new = image.copy()
	for i in range(m):
		for j in range(n):
			new[i, j] = s[new[i, j]]
	return new


def local_histogram(filename, size, step):
	try:
		img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	except IOError:
		print('no such file')
	m, n = np.shape(img)
	new = img.copy()
	row_index = -1 * step
	flag1, flag2 = 0, 0
	while flag1 <= 1:
		col_index = 0
		flag2 = 0
		if row_index + max(step, size[0]) > m - max(step, size[0]):
			row_index = m - max(step, size[0])
			flag1 += 1
		else:
			row_index += step
		while flag2 <= 1:
			l = np.max(new[row_index:row_index+size[0], col_index:col_index+size[1]])
			new[row_index:row_index+size[0], col_index:col_index+size[1]] = \
				local_cal(img[row_index:row_index+size[0], col_index:col_index+size[1]], l)
			if col_index + max(step, size[1]) > n - max(step, size[1]):
				col_index = n - max(step, size[1])
				flag2 += 1
			else:
				col_index += step
	return img, new


def segment(img):
	add = 0
	_, p = histogram(img)
	for i in range(len(p)):
		add += p[i]
		if add >= 0.7:
			T = i
			break
	new_1, new_2 = img.copy(), img.copy()
	new_1[new_1 > T] = 0
	new_2[new_2 > T] = 255
	return new_1, new_2


def draw(names, images, row, col, c='blue'):
	assert isinstance(names, list) and isinstance(images, list), 'unexpected data type'
	assert len(names) == len(images), 'dimension mismatch'
	plt.figure('SHOW')
	for i in range(len(names)):
		plt.subplot(row, col, i+1)
		plt.title(names[i])
		if isinstance(images[i], dict):
			plt.bar(images[i].keys(), images[i].values(), color=c)
		else:
			plt.imshow(images[i], cmap='gray')
	plt.show()


if __name__ == '__main__':
	#filename1 = 'C:/Users/peisen/Desktop/picture/woman.bmp'
	#filename2= 'C:/Users/peisen/Desktop/picture/woman2.bmp'
	#img = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
	#d, image, _d, new = equal(filename)
	#pz, _ = histogram(img)
	#orig, new = histogram_match(filename2, list(pz.values()))
	#hist_orig,_ = histogram(orig)
	#hist_new,_ = histogram(new)
	#draw(['original', 'original','processed','processed'], [hist_orig,orig,hist_new,new], row=2, col=2)
	#draw(['original', 'original','processed','processed'], [d, image,_d,new], row=2, col=2)
	filename = 'C:/Users/peisen/Desktop/picture/elain.bmp'
	#size = [7, 7]
	#step = 3
	#orig, new = local_histogram(filename, size, step)
	#draw(['orig', 'local equalization'], [orig, new], row=1, col=2)
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	background, obj = segment(img)
	draw(['background', 'object'], [background, obj], row=1, col=2)
