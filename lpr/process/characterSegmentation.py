import cv2


def img2gray(plate):
    #img = cv2.imread(filename)
    img_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', img_gray)
    # cv2.waitKey(0)
    return img_gray


def gray2thre(img_gray):
    # img_thre = img_gray
    # cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV, img_thre)
    # 这里得到的是灰度图转化为2值图像
    ret, img_thre = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', img_thre)
    # cv2.waitKey(0)
    return img_thre


# def find_end(start, width):
#     end = start+1
#     for col in range(start+1, width-1):
#         if()

def split(plate):
    img_gray = img2gray(plate)
    img_thre = gray2thre(img_gray)
    white = []
    black = []
    white_1 = []
    black_1 = []
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0
    white_max_1 = 0
    black_max_1 = 0

# 寻找水平距离
    for col in range(width):
        white_col = 0
        black_col = 0

        for row in range(height):
            if img_thre[row][col] == 255:
                white_col += 1
            if img_thre[row][col] == 0:
                black_col += 1

        white_max = max(white_max, white_col)
        black_max = max(black_max, black_col)

        white.append(white_col)
        black.append(black_col)
        # print(white_col)
        # print(black_col)

# 寻找竖直的距离
    for row_1 in range(height):
        white_row = 0
        black_row = 0

        for col_1 in range(height):
            if img_thre[row_1][col_1] == 255:
                white_row += 1
            if img_thre[row_1][col_1] == 0:
                black_row += 1

        white_max_1 = max(white_max, white_row)
        black_max_1 = max(black_max, black_row)

        white_1.append(white_row)
        black_1.append(black_row)
        # print(white_1)
        # print(black_1)

# 指定类型, false标识白底黑字, true表示黑底白字
    type = False
    if black_max > white_max:
        type = True

# 计算竖直边界
    count_1 = 1
    start_1 = 1
    end_1 = 2
    while count_1 < height - 2:
        count_1 += 1
        if (white_1[count_1] if type else black_1[count_1]) > (0.05 * white_max_1 if type else 0.05 * black_max_1):
            start_1 = count_1
            end_1 = start_1 + 1
            for row in range(start_1 + 1, height - 1):
                if (black_1[row] if type else white_1[row]) > (0.95 * black_max_1 if type else 0.95 * white_max_1):
                    end_1 = row
                    break

            count_1 = end_1
            if end_1 - start_1 > 8:
                break

    new_height = end_1 - start_1
    # print(str(new_height))

# 计算水平边界
    count = 1
    start = 1
    end = 2
    cjs = []
    while count < width-2:
        count += 1
        if (white[count] if type else black[count]) > (0.05*white_max if type else 0.05*black_max):
            start = count
            end = start+1
            for columu in range(start+1, width-1):
                if (black[columu] if type else white[columu]) > (0.95 * black_max if type else 0.95 * white_max):
                    end = columu
                    break

            count = end
            if end - start > 5:
                new_width = end-start
                bais = int((new_height-new_width)/2)
                # cj = img_gray[start_1:end_1, start-bais+2:end+bais-2] 目前效果待定,不确定是否采用灰度图
                cj = img_thre[start_1:end_1, start-bais+2:end+bais-2]
                cj = cv2.resize(cj, (20, 20), interpolation=cv2.INTER_CUBIC)
                cjs.append(cj)

    return cjs


if __name__ == '__main__':
    cjs = split(filename='crop.jpg')
    i = 0
    for cj in cjs:
        cv2.imshow(str(i), cj)
        cv2.waitKey(0)
        i = i+1
