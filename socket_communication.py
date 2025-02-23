import socket
import json
import numpy as np
import time  # Optional: for adding delays

def send_grasp_data(host='localhost', port=65432, grasps=None, scores=None):
    """
    Sends grasp matrices and scores to the ROS 2 publisher via socket.
    
    Parameters:
    - host (str): The hostname or IP address of the ROS 2 publisher.
    - port (int): The port number of the ROS 2 publisher.
    - grasps (list of np.ndarray): List of 4x4 grasp matrices.
    - scores (list of float): List of grasp scores.
    """
    if grasps is None or scores is None:
        raise ValueError("Grasps and scores must be provided.")

    # Prepare the data
    data = {
        "generated_grasps": [grasp.tolist() for grasp in grasps],
        "generated_scores": scores
    }

    # Serialize to JSON
    json_data = json.dumps(data)

    # Establish a socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            s.sendall(json_data.encode('utf-8'))
            print(f"Sent {len(grasps)} grasps and {len(scores)} scores to {host}:{port}")
        except ConnectionRefusedError:
            print(f"Failed to connect to {host}:{port}. Is the ROS 2 publisher running?")
        except Exception as e:
            print(f"Error sending data: {e}")

# Example usage
if __name__ == "__main__":
    # Sample grasp matrices and scores
    generated_grasps = [
        np.array([
            [0.20428572,  0.23623468, -0.94997922,  0.4227997],
            [-0.06873716, -0.96458758, -0.25464878,  0.05736987],
            [-0.97649503,  0.11731998, -0.18081337, -0.09540988],
            [0.0,         0.0,         0.0,          1.0]
        ]),
        # Add more matrices as needed
    ]
    generated_scores = [0.98, 0.99]  # Corresponding scores

    send_grasp_data(host='localhost', port=65432, grasps=generated_grasps, scores=generated_scores)
