import os

import runpod

from deploy.config import GPU_COUNT, POD_CONF

runpod.api_key = os.getenv("RUNPOD_API_KEY")


def deploy_llm(model_id=None):
    pod_conf = POD_CONF.copy()
    if model_id:
        pod_conf.update(
            {
                "docker_args": f"--model-id {model_id} --num-shard {GPU_COUNT}",
            }
        )
    pod = runpod.create_pod(**pod_conf)
    return f'https://{pod["id"]}-80.proxy.runpod.net'
