#path = './data/homework/train/00065.jpg'
base_img = Image.open("./data/homework/train/00065.jpg")
imagem = Image.open("./data/homework/partial_credit/00001.jpg")
# line below only needed for the example above
imagem = imagem.rotate(-90, expand=True)
# stop edit
# start code here upon image being input
imagem = imagem.resize(base_img.size)
imagem = cv2.cvtColor(np.array(imagem), cv2.COLOR_RGB2BGR)
img = cv2.bitwise_not(imagem)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()
img_blur = cv2.medianBlur(gray, 5)
edges = cv2.Canny(img_blur, 50, 200)
lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                        threshold=200, minLineLength=700, maxLineGap=12000)
l_lines = []
for line in lines:
    if len(l_lines) == 0:
        l_lines.append(line[0].tolist())
    else:
        orig_len = len(l_lines)
        breaker = 0
        for l in range(len(l_lines)):
            if l_lines[l][1] > line[0][1] and abs(l_lines[l][1] - line[0][1]) < 10:
                breaker = 1
                break
            elif l_lines[l][1] > line[0][1] and abs(l_lines[l][1] - line[0][1]) > 10:
                l_lines.insert(l-1, line[0].tolist())
        if orig_len == len(l_lines) and abs(l_lines[len(l_lines)-1][1] - line[0][1]) > 10 and breaker != 1:
            l_lines.append(line[0].tolist())

print(l_lines)

img_crop_list = []

iter = 0
for crop_line in l_lines:
    if len(img_crop_list) == 0:
        img_crop_list.append(im.crop((0, 0, crop_line[2], crop_line[3])))
        iter += 1
        if len(img_crop_list) + 1 == len(l_lines) + 1:
            corner_x, corner_y = im.size
            img_crop_list.append(
                im.crop((crop_line[0], crop_line[1], corner_x, corner_y)))
    elif len(img_crop_list) + 1 != len(l_lines) + 1:
        img_crop_list.append(im.crop(
            (l_lines[iter - 1][0], l_lines[iter - 1][1], crop_line[2], crop_line[3])))
        iter += 1
        if len(img_crop_list) + 1 == len(l_lines) + 1:
            corner_x, corner_y = im.size
            img_crop_list.append(
                im.crop((crop_line[0], crop_line[1], corner_x, corner_y)))
    else:
        corner_x, corner_y = im.size
        img_crop_list.append(
            im.crop((crop_line[0], crop_line[1], corner_x, corner_y)))

for crp in img_crop_list:
    plt.imshow(crp)
    plt.show()
