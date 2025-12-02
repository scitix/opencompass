"""评估脚本通用工具 - API配置加载"""
import os


def load_env_file(env_path: str = ".env") -> None:
    """
    加载 .env 文件到环境变量

    Args:
        env_path: .env文件路径，默认为当前目录下的.env
    """
    if not os.path.exists(env_path):
        return

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # 去除引号
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                # 不覆盖已有的环境变量
                if key not in os.environ:
                    os.environ[key] = value


def get_api_config():
    """
    获取API配置（从环境变量）

    Returns:
        dict: 包含base_url, model_name, api_key的字典
    """
    return {
        'base_url': os.environ.get('EVAL_BASE_URL', 'http://localhost:8000/v1'),
        'model_name': os.environ.get('EVAL_MODEL_NAME', 'default-model'),
        'api_key': os.environ.get('EVAL_API_KEY', 'EMPTY'),
    }


# 自动加载.env文件
load_env_file()
