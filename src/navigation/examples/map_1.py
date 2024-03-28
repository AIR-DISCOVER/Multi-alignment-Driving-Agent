import carla

# 建立Carla客户端连接
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

try:
    # 获取Carla世界
    world = client.load_world('Town05')

    # 创建地图点的坐标列表
    spawn_points = [
        carla.Transform(carla.Location(x=-164.732162, y=-95.141876, z=0.300000)),
        carla.Transform(carla.Location(x=-44.188328, y=-39.610611, z=0.450000)),
        # 继续添加其他的坐标
    ]

    # 在Carla世界中创建地图点
    for transform in spawn_points:
        world.debug.draw_point(transform.location, size=0.1, color=carla.Color(255, 0, 0), life_time=120)

    # 运行Carla仿真
    while True:
        pass

finally:
    print("关闭Carla客户端")
