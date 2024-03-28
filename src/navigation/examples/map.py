import carla
import random
import time

# Carla连接配置
host = "localhost"
port = 2000
client = carla.Client(host, port)
client.set_timeout(10.0)

# 获取Carla世界
world = client.get_world()

# 获取Carla地图
map = world.get_map()

# 获取地图的可生成点
spawn_points = map.get_spawn_points()

# 创建一个起点和终点
start_location = carla.Transform(carla.Location(x=100, y=100, z=2), carla.Rotation())
end_location = carla.Transform(carla.Location(x=200, y=100, z=2), carla.Rotation())

# 保存生成点的位置
spawn_point1 = random.choice(spawn_points)
spawn_point2 = random.choice(spawn_points)
spawn_point3 = random.choice(spawn_points)

# 创建路段1：无干扰路段
def create_no_disturbance_segment():
    segment = map.get_waypoint(start_location.location).next(50.0)[0]
    return segment


# 创建路段2：有信号灯和前方车辆的干扰路段
def create_signal_and_vehicle_segment():
    segment = map.get_waypoint(start_location.location).next(100.0)[0]
    # 在路段上生成一辆前方车辆
    vehicle_bp = world.get_blueprint_library().filter('vehicle.*')[0]

    vehicle_transform = carla.Transform(
        carla.Location(x=spawn_point1.location.x, y=spawn_point1.location.y, z=spawn_point1.location.z),
        carla.Rotation())
    vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
    return segment, vehicle


# 创建路段3：复杂路段，包含多个干扰因素
def create_complex_segment():
    segment = map.get_waypoint(start_location.location).next(200.0)[0]
    # 在路段上生成一辆前方车辆和一个前方行人
    vehicle_bp = world.get_blueprint_library().filter('vehicle.*')[0]

    vehicle_transform = carla.Transform(
        carla.Location(x=spawn_point2.location.x, y=spawn_point2.location.y, z=spawn_point2.location.z),
        carla.Rotation())
    vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)

    pedestrian_bp = world.get_blueprint_library().filter('walker.pedestrian.*')[0]

    pedestrian_transform = carla.Transform(
        carla.Location(x=spawn_point3.location.x, y=spawn_point3.location.y, z=spawn_point3.location.z),
        carla.Rotation())
    pedestrian = world.spawn_actor(pedestrian_bp, pedestrian_transform)

    # 手动更新行人的位置，模拟其沿着路段移动
    pedestrian_location = pedestrian.get_location()
    target_location = segment.transform.location
    delta_location = target_location - pedestrian_location
    delta_location = delta_location / delta_location.distance(target_location)

    # 每帧更新行人的位置
    while pedestrian_location.distance(target_location) > 2.0:
        pedestrian_location = pedestrian.get_location()
        pedestrian_location.x += delta_location.x
        pedestrian_location.y += delta_location.y
        pedestrian.set_location(pedestrian_location)
        world.tick()  # 更新世界状态
        time.sleep(0.05)  # 控制更新速度

    return segment, vehicle, pedestrian

# 创建路段
segment1 = create_no_disturbance_segment()
segment2, vehicle2 = create_signal_and_vehicle_segment()
segment3, vehicle3, pedestrian3 = create_complex_segment()

# 打印生成点的位置
print("生成点1位置:", spawn_point1.location)
print("生成点2位置:", spawn_point2.location)
print("生成点3位置:", spawn_point3.location)

# 设置导航起点和终点
start_location = segment1.transform
end_location = segment3.transform

# 在终点附近生成一个目标点，用于结束导航
end_waypoint = map.get_waypoint(end_location.location)

try:
    # 创建车辆
    vehicle_bp = world.get_blueprint_library().filter('vehicle.*')[0]

    vehicle_transform = carla.Transform(
        carla.Location(x=spawn_point1.location.x, y=spawn_point1.location.y, z=spawn_point1.location.z),
        carla.Rotation(yaw=0))
    vehicle1 = world.spawn_actor(vehicle_bp, vehicle_transform)

    # 设置车辆的初始位置
    vehicle1.set_transform(start_location)

    # 开始车辆的自动驾驶
    vehicle1.set_autopilot(True)

    # 等待导航完成
    while True:
        location = vehicle1.get_location()
        if location.distance(end_waypoint.transform.location) < 2.0:
            print("导航完成")
            break

except Exception as e:
    print(f"生成角色时出错：{e}")

finally:
    # 销毁所有角色
    if 'vehicle1' in locals():
        vehicle1.destroy()
    if 'vehicle2' in locals():
        vehicle2.destroy()
    if 'vehicle3' in locals():
        vehicle3.destroy()
    if 'pedestrian3' in locals():
        pedestrian3.destroy()

    world.wait_for_tick()
