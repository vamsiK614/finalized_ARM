import cv2


def letterbox(img, new_shape=640, color=(114, 114, 114)):

    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)

    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded, scale, left, top
