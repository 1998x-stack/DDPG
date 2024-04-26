import gym, json, re

def get_all_envs():
    # 获取所有Gym环境的名称列表
    env_names = list(gym.envs.registry.keys())

    js_env = {}

    # 遍历所有环境，尝试创建并获取环境信息
    for env_name in env_names[::-1]:
        env = None
        try:
            # 创建环境
            env = gym.make(env_name)
            # 获取观察空间和行动空间
            obs_space = env.observation_space
            action_space = env.action_space

            # 初始化观察空间和行动空间的描述字符串
            obs_space_desc = ""
            action_space_desc = ""
            obs_space_discrete = False
            action_space_discrete = False
            min_action = None
            max_action = None
            
            # 格式化观察空间描述
            if isinstance(obs_space, gym.spaces.Discrete):
                obs_space_desc = obs_space.n
                obs_space_discrete = True
            elif isinstance(obs_space, gym.spaces.Box):
                obs_space_desc = obs_space.shape if len(obs_space.shape) > 1 else obs_space.shape[0]

            # 格式化行动空间描述
            if isinstance(action_space, gym.spaces.Discrete):
                action_space_desc = action_space.n
                action_space_discrete = True
            elif isinstance(action_space, gym.spaces.Box):
                action_space_desc = action_space.shape if len(action_space.shape) > 1 else action_space.shape[0]
                min_action = action_space.low.tolist()   # 获取最小动作值
                max_action = action_space.high.tolist()  # 获取最大动作值
                
            # 打印环境名称和相关信息
            print(f"环境名称: {env_name}")
            print(f"观察空间: {obs_space_desc}")
            print(f"行动空间: {action_space_desc}")
            print(f"观察空间离散: {obs_space_discrete}")
            print(f"行动空间离散: {action_space_discrete}")
            # 打印动作空间的最小和最大值（如果它们存在的话）
            if min_action is not None and max_action is not None:
                print(f"最小动作: {min_action}, 最大动作: {max_action}")
            print("-" * 40)  # 输出分隔线以增强可读性
            js_env[env_name] = {
                "observation_space": obs_space_desc,
                "action_space": action_space_desc,
                "observation_space_discrete": obs_space_discrete,
                "action_space_discrete": action_space_discrete,
                "action_range": [min_action, max_action] if min_action is not None and max_action is not None else []
            }

        except gym.error.Error as e:
            # Gym特定错误处理
            print(f"无法加载环境 {env_name}: {e}")
        except Exception as e:
            # 其他错误处理
            print(f"在尝试创建环境 {env_name} 时发生错误: {e}")

        finally:
            # 确保环境被正确关闭
            if env:
                env.close()

    with open("data/env.json", "w") as f:
        json.dump(js_env, f, ensure_ascii=False, indent=4)
    
    return js_env

def get_latest_envs():
    with open("data/env.json", "r") as f:
        envs = json.load(f)
    
    latest_envs = {}
    for env in envs:
        # env is like xxx-vx, use regrex
        env_name = re.match(r"(.+)-v\d+", env).group(1)
        env_version = int(re.match(r".+-(v\d+)", env).group(1)[1:])
        
        latest_envs[env_name] = max(env_version, latest_envs.get(env_name, 0))
    
    ret_json = {}
    for env_name, env_version in latest_envs.items():
        ret_json[f"{env_name}-v{env_version}"] = envs[f"{env_name}-v{env_version}"]
        
    with open("data/latest_env.json", "w") as f:
        json.dump(ret_json, f, ensure_ascii=False, indent=4)
    
    return ret_json
        
if __name__ == '__main__':
    get_all_envs()
    get_latest_envs()
    pass