import cv2

# Запись движения на видео
def definitionVideoMotions():
    # Чтение исходного видео
    video = cv2.VideoCapture("video.mov")
    video.set(cv2.CAP_PROP_FPS, 60)  # Частота кадров

    # Чтение кадра
    ok, frame = video.read()
    # Инициализация чёрно-белого изображения
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Размытие по Гауссу
    blurryFrame = cv2.GaussianBlur(grayFrame, (5, 5), 1)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Параметры записи видео
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w,h))
    while (True):
        oldFrame = blurryFrame
        # Чтение следующего кадра
        ok, frame = video.read()

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurryFrame = cv2.GaussianBlur(grayFrame, (5, 5), 1)

        # Разница между двумя кадрами
        frame_diff = cv2.absdiff(oldFrame, blurryFrame)

        # Операция двоичного разделения
        ret_val, frame_thresh = cv2.threshold(frame_diff, 127, 255, cv2.THRESH_BINARY)

        # Нахождение контуров объектов
        contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Находим контуры, площадью больше заданного параметра
        for i in contours:
            if 300 > cv2.contourArea(i):
                cv2.imshow('img', frame)
                # Записываем кадр в результирующее видео
                video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def main():
   definitionVideoMotions()


if __name__ == '__main__':
    main()