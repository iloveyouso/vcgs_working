import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import ros2_numpy  # ROS 2용 ros_numpy 라이브러리
import os
import datetime

#this is for graspnet


class PointCloudSaver(Node):
    def __init__(self):
        super().__init__('pointcloud_saver')
        
        # 구독할 토픽 이름을 설정하세요
        topic_name = '/filtered_points'
        
        # QoS 설정 (필요에 따라 조정 가능)
        qos_profile = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        
        # PointCloud2 토픽 구독
        self.subscription = self.create_subscription(
            PointCloud2,
            topic_name,
            self.pointcloud_callback,
            qos_profile
        )
        self.subscription  # prevent unused variable warning

        self.get_logger().info(f'Subscribed to {topic_name}')

    def pointcloud_callback(self, msg):
        self.get_logger().info('PointCloud2 메시지 수신, 저장 중...')

        # PointCloud2 데이터를 numpy 배열으로 변환
        smoothed_object_pc = ros2_numpy.point_cloud2.point_cloud2_to_array(msg)
        
        # .npy에 저장할 dictionary 생성
        data_dict = {
            'smoothed_object_pc': smoothed_object_pc,
            'depth': 'dummy_depth',  # 더미 값
            'base_to_camera_rt': 'base_to_camera_rt',  # 문자열 값
            'image': 'dummy_image',  # 더미 값
            'intrinsics_matrix': 'dummy_intrinsics_matrix'  # 더미 값
        }
        
        # 파일 저장 경로 설정 (필요에 따라 변경 가능)
        file_name = f"pointcloud_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        
        # .npy 파일로 저장
        np.save(save_path, data_dict)
        self.get_logger().info(f'PointCloud2 데이터가 {save_path}에 저장되었습니다.')

        # 한 번 저장 후 노드 종료 (필요에 따라 제거 가능)
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    pointcloud_saver = PointCloudSaver()
    try:
        rclpy.spin(pointcloud_saver)
    except KeyboardInterrupt:
        pass
    finally:
        pointcloud_saver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
