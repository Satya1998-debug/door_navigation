# Door detection from YOLO
std_msgs/Header header
float32[] bbox          # [x1, y1, x2, y2] in image coordinates
int32 class_id          # 0=door_double, 1=door_single, 2=handle
float32 confidence
