
import os
import json
import mlflow
import mlflow.lightgbm
import mlflow.sklearn

def log_mlflow(experiment_name, model, params, metrics, tags=None, image_local_path=None, artifacts=None, config=None, track_ui=None):
    """
    记录模型和实验数据到 MLflow。

    参数:
    - experiment_name (str): 实验的名称。
    - model: 训练得到的模型对象。
    - params (dict): 模型的参数。
    - metrics (dict): 实验的指标。
    - tags (dict, optional): 标签字典。默认值为 None。
    - image_local_path (str, optional): 本地图片路径。默认值为 None。
    - artifacts (dict, optional): 其他要记录的文件或数据（例如配置文件等）。默认值为 None。
    - config (dict, optional): 要记录的配置内容（假设是 JSON 格式）。默认值为 None。
    - track_ui (str, optional): MLflow 跟踪 URI。如果提供将设置为此 URI。默认值为 True。

    返回:
    - mlflow run ID
    """

    # 如果有活动的 run，先结束它
    if mlflow.active_run():
        mlflow.end_run()

    # 设置 MLflow UI 跟踪 URI
    if track_ui:
        mlflow.set_tracking_uri(track_ui)

    # 启动 mlflow 实验
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # 记录模型
        try:
            mlflow.sklearn.log_model(model, 'model')

        except Exception as e:
            print(f"Error logging model: {e}")
            raise

        # 记录模型参数
        if params:
            mlflow.log_params(params)

        # 记录实验指标
        if metrics:
            mlflow.log_metrics(metrics)

        # 记录 tags
        if tags:
            mlflow.set_tags(tags)

        # 记录配置内容
        if config:
            mlflow.log_text(json.dumps(config), "config.json")  # 将配置内容记录为 JSON 文件

        # 记录其他文件或数据
        if artifacts:
            for local_path, artifact_path in artifacts.items():
                mlflow.log_artifact(local_path, artifact_path)

        # 记录图片
        if image_local_path:
            mlflow.log_artifact(image_local_path, 'model')

        # 获取 mlflow run ID
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run ID: {run_id}")

    return run_id

def load_model_and_config(remote_tracking_uri, run_id):
    """
    从远程 MLflow Tracking Server 获取模型和配置文件内容。

    参数:
    - remote_tracking_uri (str): 远程服务器的 Tracking URI。
    - run_id (str): 要获取的运行 ID。

    返回:
    - model: 加载的模型对象。
    - config: 加载的配置文件内容（假设是 JSON 格式）。
    """

    # 设置远程服务器的 Tracking URI
    mlflow.set_tracking_uri(remote_tracking_uri)

    # 参数验证
    if not isinstance(remote_tracking_uri, str) or not remote_tracking_uri:
        raise ValueError("remote_tracking_uri must be a non-empty string.")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string.")

    # 获取运行信息
    try:
        run_info = mlflow.get_run(run_id).info
    except Exception as e:
        print(f"Error fetching run info: {e}")
        raise

    # 使用 MLflow API 加载模型
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # 获取配置文件内容
    try:
        config_artifact_path = os.path.join(run_info.artifact_uri, "config.json")
        config_content = mlflow.artifacts.download_artifacts(config_artifact_path)
        with open(config_content, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

    return model, config