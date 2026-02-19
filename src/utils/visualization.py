import cv2


def draw_detections(frame, boxes, confidences, class_ids, class_names):

    for box, conf, cls_id in zip(boxes, confidences, class_ids):

        label_name = class_names[cls_id]
        x, y, w, h = box
        label = f"{label_name}: {conf:.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame
