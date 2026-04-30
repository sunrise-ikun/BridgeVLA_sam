import random
import socket
import re
import time
import os
import threading 
import pickle
import numpy as np
from datetime import datetime
from pynput import keyboard
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2mat
import transforms3d
return_to_initial_pose = False
replay_motion = False
claw_open = False
claw_close = False  
set_drag = False
reset_drag = False

class Server():
    def __init__(self, ip, port, host, app_port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.app = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp = None
        self.host = host
        self.app_port = app_port
        self.baudrate = 115200
        self.modbus = None
        self.modbusRTU = None
        self.timestamp = None

    def init_com(self):
        self.modbus = self.get_id(self.modbus)
        self.modbusRTU = self.get_id(self.modbusRTU)

    def get_id(self, response):
        match = re.search(r'\{(.+?)\}', response)
        if match:
            numbers_str = match.group(1)
            return int(numbers_str)

    def start_server(self):
        """启动服务器"""
        self.app = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.app.bind((self.host, self.app_port))
        self.app.listen(5)
        print(f"Server listening on {self.host}:{self.app_port}")

        try:
            while True:
                self.tcp, addr = self.app.accept()
                print(f"Accepted connection from {addr[0]}:{addr[1]}")
                client_handler = threading.Thread(
                    target=self.handle_client,
                    args=(self.tcp, )
                )
                client_handler.daemon = True
                client_handler.start()
        except KeyboardInterrupt:
            print("Stopping server...")
            self.app.close()
            self.tcp.close()

    def handle_client(self,sock):
        """处理客户端的通信"""
        global return_to_initial_pose
        global replay_motion
        global claw_open
        global claw_close
        try:
            while True:
                data = sock.recv(1024).decode().strip()
                if not data:
                    print("Client disconnected")
                    break
                print(f"Received data: {data}")
                if data == 'replay':
                    replay_motion = True
                elif data == 'back':
                    return_to_initial_pose = True
                elif data == 'open':
                    claw_open = True
                elif data == 'close':
                    claw_close = True
                sock.send("Message received".encode())
        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            sock.close()

class Point():
    def __init__(self, name:str, position: dict):
        self.name = name
        self.position = position
        self.timestamp = None
        self.position_str = None
        self.claw = None
        self.position_to_string()

    def get_timestamp(self):
        dt = datetime.now()
        micro = dt.microsecond // 1000
        self.timestamp = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}")
        return self.timestamp

    def position_to_string(self):
        if self.position == None:
            return None
        self.position_str = f"{{{self.position['x']:.4f},{self.position['y']:.4f},{self.position['z']:.4f},\
            {self.position['rx']},{self.position['ry']},{self.position['rz']}}}"
        return self.position_str


def find_latest_number_folder(folder_path):
    """查找最大数字命名的文件夹"""
    if not os.path.exists(folder_path):
        return None
        
    dir_list = os.listdir(folder_path)
    number_folders = []
    
    for name in dir_list:
        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            number_folders.append(int(name))
    
    if not number_folders:
        return None
    
    # 返回最大数字对应的文件夹名
    max_number = max(number_folders)
    max_number=0#gai
    return str(max_number)


def wait_and_prompt(sock, point=None, state = True, replay = False):
    while get_status(sock) not in [5, 6]:
        time.sleep(0.1)
    if not state:
        user_input = input("机械臂空闲，输入''继续下一动作：")
        while user_input.lower() != '':
            print("输入无效，请输入''确认继续！")
            user_input = input("机械臂空闲，输入''继续下一动作：")

ptest = Point('ptest', None)

def replay_motion_trajectory(sock, modbus, timestamp, replay=True):
    trajectory_points = []
    gripper_states = []  # 存储夹爪状态
    global ptest
    print("检测到轨迹文件，输入''确认执行复现，其他输入取消：")
    if replay:
        user_choice = input().lower()
        if user_choice != '':
            print("取消轨迹复现，继续等待'q'键...")
            return 0
    
    try:
        # 查找最大数字命名的文件夹
        base_folder = '/home/zk/Projects/Datasets/put_the_coke_can_in_the_bottom_shelf_stand_0819'  #  采集数据集保存路径  动物进抽屉
        # base_folder = '/home/zk/Projects/Datasets/put_block_on_the_plate_0604/'  #方块
        # base_folder = '/home/zk/Projects/Datasets/put_the_bottle_in_the_microwave_25_0604/'  #微波炉
        latest_folder = find_latest_number_folder(base_folder)
        
        if latest_folder is None:
            print("未找到数字命名的文件夹")
            return 0
        
        actions_folder = os.path.join(base_folder, latest_folder, 'actions')
        
        if not os.path.exists(actions_folder):
            print(f"actions文件夹不存在: {actions_folder}")
            return 0
        
        # 获取所有pkl文件并按数字排序
        pkl_files = []
        for filename in os.listdir(actions_folder):
            if filename.endswith('.pkl') and filename[:-4].isdigit():
                pkl_files.append((int(filename[:-4]), filename))
        
        pkl_files.sort(key=lambda x: x[0])  # 按数字排序
        
        # 读取每个pkl文件
        for number, filename in pkl_files:
            pkl_path = os.path.join(actions_folder, filename)
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                # 数据格式为 [x, y, z, qx, qy, qz, qw, gripper_state]
                if len(data) >= 8:
                    x, y, z = data[0]*1000, data[1]*1000, data[2]*1000
                    # qx, qy, qz, qw = data[3], data[4], data[5], data[6]
                    pose_quat = data[3:7]
                    gripper_state = data[7]  # 夹爪状态
                    pose_eurl = transforms3d.euler.quat2euler(pose_quat, axes='sxyz')
                    pose_eurl = np.rad2deg(np.asarray(pose_eurl))
                    # 将四元数转换为欧拉角
                    rx = float(pose_eurl[0])  
                    ry = float(pose_eurl[1])
                    rz = float(pose_eurl[2])
                    
                    
               
                    trajectory_points.append({
                        'x': x, 'y': y, 'z': z,
                        'rx': rx, 'ry': ry, 'rz': rz
                    })
                    gripper_states.append(gripper_state)
                else:
                    print(f"警告：{filename} 数据格式不正确（需要8个数据），跳过")
            except Exception as e:
                print(f"读取 {filename} 时出错: {e}")
        
        
        
        if len(trajectory_points) > 0:
            print(f"开始轨迹复现：{actions_folder}")
            last_gripper_state = None  # 记录上一个夹爪状态，避免重复操作
            
            for i, point in enumerate(trajectory_points):
                point_str = f"{{{point['x']:.4f},{point['y']:.4f},{point['z']:.4f},{point['rx']:.4f},{point['ry']:.4f},{point['rz']:.4f}}}"
                send_movj_command(sock, point_str)
                wait_and_prompt(sock, state=True, replay=False)
                
                # 检查夹爪状态是否发生变化
                if i < len(gripper_states):
                    current_gripper_state = gripper_states[i]
                    if last_gripper_state is None or current_gripper_state != last_gripper_state:
                        print(f"夹爪状态变化: {last_gripper_state} -> {current_gripper_state}")
                        if current_gripper_state == 1:
                            print("关闭夹爪")
                            claws_control(sock, 0, modbus, ptest)  # 0表示关闭
                        elif current_gripper_state == 0:
                            print("打开夹爪")
                            claws_control(sock, 1, modbus, ptest)  # 1表示打开
                        last_gripper_state = current_gripper_state
                
            print("轨迹复现完成！")
        else:
            print("没有找到有效的轨迹点")
    except Exception as e:
        print(f"轨迹复现出错: {e}")

def main():
    global return_to_initial_pose
    global replay_motion
    global claw_open
    global claw_close
    global set_drag
    global reset_drag
    global lift_before_return  # 新增标志位
    global only1
    server = Server('192.168.201.1', 29999, '192.168.201.38', 12345)
    # print(server.port)
    server.sock.connect((server.ip, server.port))
    initialize_robot(server.sock)
    server_modbus = 'ModbusCreate("192.168.201.1", 502,2)'
    server_modbusrtu = 'ModbusRTUCreate(1, 115200, "N", 8, 1)'
    server.modbus = send_modbus_command(server.sock, server_modbus)
    server.modbusRTU = send_modbus_command(server.sock,server_modbusrtu)
    server.init_com()
    
    p1 = Point('p1', {'x': 173.2972, 'y': -130.0664, 'z': 167.6071, 'rx': 91.3785, 'ry': -0.9884, 'rz': -67.1547})
    p2 = Point('p2', {'x': 168.5996, 'y': -223.9824, 'z': 60, 'rx': 90.1654, 'ry': -3.2263, 'rz': -59.2746})
    p3 = Point('p3', {'x': -77.1242, 'y': -610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p4 = Point('p4', {'x': -187.2389, 'y':-610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p5 = Point('p5', {'x': -77.1242, 'y': -610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p6 = Point('p6', {'x': 168.5996, 'y': -223.9824, 'z': 60, 'rx': 90.1654, 'ry': 3.2263, 'rz': -59.2746})
    p7 = Point('p7', {'x': 173.2972, 'y': -130.0664, 'z': 167.6071, 'rx': 91.3785, 'ry': -0.9884, 'rz': -67.1547})
    point_list = [p1, p2, p3, p4, p5, p6, p7]
    
    def input_listener():
        global return_to_initial_pose
        global replay_motion
        global claw_open
        global claw_close
        global set_drag
        global reset_drag
        global lift_before_return
        global only1
        while True:
            try:
                key = input().strip()
            except EOFError:
                break
            if key == 'q':
                replay_motion = True
            elif key == 'i':  # 直接返回初始位置
                return_to_initial_pose = True
                lift_before_return = False
            elif key == 'u':  # 先抬高再返回初始位置
                return_to_initial_pose = True
                lift_before_return = True
            elif key == 'e':
                claw_close = True
            elif key == 'r':
                claw_open = True
            elif key == 'o':
                set_drag = True
            elif key == 'p':
                reset_drag = True
            elif key == 'k':  # 
                return_to_initial_pose = True
                only1 = True
    listener = threading.Thread(target=input_listener, daemon=True)
    listener.start()
    
    try:
        while True:
            print("按键说明:")
            print("'q' - 轨迹复现")
            print("'i' - 直接返回初始位置")
            print("'u' - 先抬高再返回初始位置")
            print("'e' - 夹爪关闭")
            print("'r' - 夹爪打开")
            print("'o' - 设置拖动")
            print("'p' - 重置拖动")
            print("'k' - ts")
            while not (replay_motion or return_to_initial_pose or claw_open or claw_close or set_drag or reset_drag):
                pass
            print("开始机械臂运动...")
            
            dt = datetime.now()
            micro = dt.microsecond // 1000
            timestamp_start = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}_r_wbl")
            
            if replay_motion:
                replay_motion_trajectory(server.sock, server.modbusRTU, timestamp_start)
            elif return_to_initial_pose:
                if lift_before_return:  # 如果需要先抬高
                    # 获取当前位置
                    current_pose = send_command(server.sock, "GetPose()")
                    try:
                        # 从响应中提取坐标数据
                        pose_data = re.search(r'\{(.+?)\}', current_pose).group(1)
                        x, y, z, rx, ry, rz = map(float, pose_data.split(','))
                        
                        # 抬高
                        lift_pose = f"{{{x+20:.4f},{y:.4f},{z+150:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
                        send_movj_command(server.sock, lift_pose)
                        wait_and_prompt(server.sock)
                        current_pose = send_command(server.sock, "GetPose()")
                        pose_data = re.search(r'\{(.+?)\}', current_pose).group(1)
                        x, y, z, rx, ry, rz = map(float, pose_data.split(','))
                        lift_pose = f"{{{x+100:.4f},{y-110:.4f},{z:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
                        send_movj_command(server.sock, lift_pose)
                        wait_and_prompt(server.sock)
                        print("已抬高，继续返回初始位置")
                    except Exception as e:
                        print(f"获取或解析当前位置出错: {e}")
        
                # if only1 :
                #     # 获取当前位置
                #     current_pose = send_command(server.sock, "GetPose()")
                #     try:
                #         # 从响应中提取坐标数据
                #         pose_data = re.search(r'\{(.+?)\}', current_pose).group(1)
                #         x, y, z, rx, ry, rz = map(float, pose_data.split(','))
                        
                #         # 抬高200mm
                #         lift_pose = f"{{{x:.4f},{y-130:.4f},{z:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
                #         send_movj_command(server.sock, lift_pose)
                #         wait_and_prompt(server.sock)
                #         # current_pose = send_command(server.sock, "GetPose()")
                #         # pose_data = re.search(r'\{(.+?)\}', current_pose).group(1)
                #         # x, y, z, rx, ry, rz = map(float, pose_data.split(','))
                #         # lift_pose = f"{{{x:.4f},{y:.4f},{z+200:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
                #         # send_movj_command(server.sock, lift_pose)
                #         # wait_and_prompt(server.sock)
                #         # print("已抬高200mm，继续返回初始位置")
                #     except Exception as e:
                #         print(f"获取或解析当前位置出错: {e}")
            
                # 返回初始位置
                # initial_pose = [210, 0.79, -81, -1.9 ,89, 37]
                initial_pose = [224, 21, -87, -18 ,88, 37]
                # initial_pose = [-122, 41.6, -118, 14.3 ,60, 52]
                # initial_pose = [238.2, 22, -76.6, -23.4 ,83, 56]
                # initial_pose = [256, 35, -101, -9.68 ,51, 76.7]

                valid_length = np.arange(0, 10, 0.1)
                valid_deg = np.arange(0, 1 ,0.1)
                initial_pose = [f"{num:.4f}" for num in initial_pose]
                initial_pose_str = f"{{{','.join(initial_pose)}}}"
                send_movjoint_commad(server.sock, initial_pose_str)
                time.sleep(2)
                
            elif claw_close:
                claws_control(server.sock, 0, server.modbusRTU, p2)
                claws_control_degree(server.sock, server.modbusRTU, 0, None)
            elif claw_open:
                claws_control(server.sock, 1, server.modbusRTU, p4)
            elif set_drag:
                send_command(server.sock, 'StartDrag()')
            elif reset_drag:
                send_command(server.sock,'ClearError()')
            
            print("机械臂运动完成。")
            
            replay_motion = False
            return_to_initial_pose = False
            claw_open = False
            claw_close = False
            set_drag = False
            reset_drag = False
            
    finally:
        close_modbus(server.sock)
        server.sock.close()
        server.app.close()
        pass  # listener thread is daemon, will exit automatically

def initialize_robot(sock):
    send_command(sock, "PowerOn()")
    time.sleep(1)
    send_command(sock, "EnableRobot()")
    send_command(sock, "ClearError()")

def send_command(sock, command):
    sock.sendall(f"{command}\n".encode('utf-8'))
    response = sock.recv(1024).decode('utf-8')
    print(f"Command: {command}")
    print(f"Response: {response}")
    return response

def send_movj_command(sock, point):
    command = f"MovJ(pose={point},a=30,v=30)"
    send_command(sock, command)

def send_movjoint_commad(sock,point):
    command = f"MovJ(joint={point},a=30,v=30)"
    send_command(sock, command)

def send_modbus_command(sock, command):
    command = f"{command}"
    return send_command(sock, command)

def claws_send_command(sock, id, num1, num2, num3):
    command = f'SetHoldRegs({id}, {num1}, {num2}, {{{num3}}}, "U16")'
    send_command(sock, command)

def claws_control(sock, status, id, point):
    if status:
        claws_send_command(sock, id, 258, 1, 0)
        claws_send_command(sock, id, 259, 1, 1)
        claws_send_command(sock, id, 264, 1, 1)
        claws_send_command(sock, 0, 258, 1, 0)
        time.sleep(1)
        point.claw = 0
    else:
        claws_send_command(sock, id, 258, 1, 1)
        claws_send_command(sock, id, 259, 1, 0)
        claws_send_command(sock, id, 264, 1, 1)
        claws_send_command(sock, 0, 258, 1, 1) 
        time.sleep(1)
        point.claw = 1
def claws_control_degree(sock, id, set_degree, point):
        if point != None:
            point.claw = set_degree
        if set_degree < 0:
            set_degree = 0
        if set_degree > 100:
            set_degree = 100
        control_value = int(9000 - set_degree * 9000 / 100) # 传入角度线性转换成控制参数

        claws_send_command(sock, id, 258, 1, 0)
        claws_send_command(sock, id, 259, 1, control_value)
        claws_send_command(sock, id, 264, 1, 1)
        time.sleep(1)
def close_modbus(sock):
    for index in range(4):
        send_modbus_command(sock, f'Modbusclose({index})')
        send_modbus_command(sock, f'Modbusclose({index})')
def get_status(sock):
    command = "RobotMode()"
    response = send_command(sock, command)
    status_code = int(response.split(',')[1][1])
    return status_code

if __name__ == "__main__":
    main()