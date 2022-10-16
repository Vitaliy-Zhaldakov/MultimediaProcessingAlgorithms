import math

import cv2
import numpy as np

def gaussMethod(n, sigma, img):

    # Копия картинки
    newImage = img

    # Параметры картинки
    height = len(img)
    width = len(img[0])

    # Центр свертки
    a = b = n // 2

    # Сумма элементов матрицы свертки
    sum = 0

    # Инициализация матрицы
    gauss_matrix = [[0] * n for i in range(n)]

    # Заполнение матрицы свертки значениями функции Гаусса с мат. ожиданием, равным координатам центра матрицы
    for x in range(n):
        for y in range(n):
            gauss_matrix[x][y] = (1 / (2 * np.pi * sigma ** 2 )) * np.exp (-((((x - a) ** 2 ) + ((y - b) ** 2)) / 2 * sigma ** 2) )
            sum += gauss_matrix[x][y]

    # Нормирование матрицы
    for x in range(n):
        for y in range(n):
            gauss_matrix[x][y] /= sum

    # Границы обхода
    height_start = a
    width_start = a
    height_finish = height - a
    width_finish  = width - a

    # Запись нового значения насыщенности каждому внутреннему пикселю
    for i in range(height_start,height_finish):
        for j in range(width_start, width_finish):
            newVal = 0
            for k in range(n):
                for l in range(n):
                    newVal = newVal + gauss_matrix[k][l] * newImage[i - a + k][j - a + l]
            newImage[i][j] = newVal

    # Размытая картинка
    cv2.namedWindow('Blurry picture')
    cv2.imshow('Blurry picture', newImage)

# Использование встроенного Гаусс-метода
def built_in_method(size, sigma, img):

    # Встроенный метод
    blurryImg = cv2.GaussianBlur(img, (size, size), sigma)
    cv2.namedWindow('Built-in blur')
    cv2.imshow('Built-in blur', blurryImg)

    # Ручной метод
    blur = gaussMethod(size, sigma, img)

def main():
    img = cv2.imread(r'ruby.png', cv2.IMREAD_GRAYSCALE)
    # Исходная картинка
    cv2.namedWindow('Default picture')
    cv2.imshow('Default picture', img)

    n = 5
    sigma = 1
    gaussMethod(n, sigma, img)
    #built_in_method(n, sigma, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()