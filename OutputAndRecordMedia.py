import cv2

# Отображение картинки
def pictureLoading():
    img = cv2.imread(r'img.jpg')

    # Создание окна, управление режимами отрисовки
    cv2.namedWindow('Picture', cv2.WINDOW_NORMAL)
    # Отображение изображения
    cv2.imshow('Picture', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Отображение видео
def videoLoading():
    video = cv2.VideoCapture(r'topVideo.mp4')
    while True:
        # Получаем результат и кадр
        ret, frame = video.read()
        if not (ret):
            break
        # Отображение кадра
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    video.release()
    cv2.destroyAllWindows()

# Запись видео с камеры
def readIPWriteTOFile():
    video = cv2.VideoCapture("http://192.168.43.1:8080/video")
    video.set(cv2.CAP_PROP_FPS, 30)  # Частота кадров
    ok, img = video.read()

    # Размеры видео
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w,h))
    while (True):
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def main():
    pictureLoading()


if __name__ == '__main__':
    main()