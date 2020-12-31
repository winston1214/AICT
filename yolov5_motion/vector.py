import numpy as np
import math

def calc_vector(point_data, fps):
    # point_data: 입력받은 바운딩 박스 좌표 array
    # bboxP1_arr: 바운딩 박스의 좌상단 좌표 array, bboxP2_arr: 바운딩 박스의 우하단 좌표 array
    # center_arr: 바운딩 박스의 중심점 좌표 array
    # next_center_arr: 현재 프레임 기준 다음 프레임의 바운딩 박스 중심점 좌표 array. 마지막 좌표값은 예측값
    # next_bboxP2_arr: 현재 프레임 기준 다음 프레임의 바운딩 박스 우하단 좌표 array. 마지막 좌표값은 예측값
    bboxP1_arr = np.array(point_data[:, :2])
    bboxP2_arr = np.array(point_data[:, 2:])
    center_arr = np.array((bboxP1_arr + ((bboxP2_arr - bboxP1_arr) / 2)))
    center_len = len(center_arr)
    next_center_arr = np.array(center_arr[1:, :])
    next_bboxP2_arr = np.array(bboxP2_arr[1:, :])
    vector_result = np.empty((0, 3), dtype=float)
    camera_distance = list()
    total_deg = 0
    total_distance = 0

    # dotvector: 두 점의 벡터 내적, deg: 수평축으로부터 벡터의 각도, distance: 두 점 사이의 거리
    # vector_result: n * 3 형태의 2차원 array. 거리, 각도, 속도 순으로 저장
    # distance_mm: 카메라와 객체와의 거리, camera_distance: distance_mm 값을 m 단위로 변환해서 저장
    for i in range(center_len):
        distance_mm = (1720 * 8) / ((bboxP2_arr[i][1] - bboxP1_arr[i][1]) * 0.005)
        camera_distance.append(distance_mm / 1000)
        if i == (center_len - 1):
            break
        dotvector = np.dot(center_arr[i], center_arr[i+1])
        theta = math.atan2(center_arr[i][1], center_arr[i][0])
        deg = theta * 180 /math.pi
        if deg < 0:
            deg += 360
        total_deg += deg
        distance = math.sqrt(math.pow(center_arr[i+1][0] - center_arr[i][0], 2)
                              + math.pow(center_arr[i+1][1] - center_arr[i][1], 2))
        total_distance += distance
        velocity = distance / (1/fps)
        vector_result = np.append(vector_result, np.array([[distance, deg, velocity]]), axis=0)

    avg_deg = total_deg / (center_len - 1)
    avg_distance = total_distance / (center_len - 1)
    final_predict_center_px = center_arr[center_len-1][0] + avg_distance * math.cos(math.radians(avg_deg))
    final_predict_center_py = center_arr[center_len-1][1] + avg_distance * math.sin(math.radians(avg_deg))
    next_center_arr = np.append(next_center_arr, np.array([[final_predict_center_px, final_predict_center_py]]), axis=0)
    final_predict_P2x = bboxP2_arr[center_len-1][0] + avg_distance * math.cos(math.radians(avg_deg))
    final_predict_P2y = bboxP2_arr[center_len-1][1] + avg_distance * math.sin(math.radians(avg_deg))
    next_bboxP2_arr = np.append(next_bboxP2_arr, np.array([[final_predict_P2x, final_predict_P2y]]), axis=0)

    return vector_result,bboxP1_arr, bboxP2_arr, center_arr,next_center_arr, next_bboxP2_arr, camera_distance

if __name__ == "__main__":
    # input_data = np.array([[3, 1, 5, 4], [6, 3, 9, 12], [9, 7, 14, 18], [11, 10, 17, 22], [15, 14, 20, 25]])
    input_data = np.array([[174,139,266,453],[151,137,264,462],[127,139,237,472],[117,143,231,480],[98,141,262,496],
                           [80,153,256,494],[80,143,253,515],[127,141,258,531],[119,151,284,529], [58,182,211,848]])
    print(calc_vector(input_data, 6))
