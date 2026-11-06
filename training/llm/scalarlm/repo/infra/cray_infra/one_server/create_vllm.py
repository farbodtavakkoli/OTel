import inspect
import os
import shlex
import sys
import torch

from cray_infra.util.get_config import get_config
from cray_infra.huggingface.get_hf_token import get_hf_token

from vllm.entrypoints.openai.api_server import build_app, \
    init_app_state, setup_server, build_async_engine_client
try:
    from vllm.entrypoints.openai.api_server import decorate_logs, get_uvicorn_log_config
except ImportError:
    from vllm.utils.system_utils import decorate_logs
    from vllm.entrypoints.launchers.utils.server_utils import get_uvicorn_log_config

from vllm.tool_parsers import ToolParserManager
try:
    from vllm.entrypoints.launcher import serve_http
except ImportError:
    from vllm.entrypoints.launchers.launcher import serve_http

try:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
except ImportError:
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

import vllm.envs as envs

import uvicorn
import logging

logger = logging.getLogger(__name__)

from cray_infra.one_server.vllm_cli_args import build_vllm_cli_args

async def create_vllm(server_status, port):
    print(f"DEBUG: BEFORE CONFIG - Environment variables:")
    print(f"  VLLM_TARGET_DEVICE: {os.environ.get('VLLM_TARGET_DEVICE', 'NOT SET')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()

    config = get_config()

    if torch.cuda.is_available():
        sm_version = torch.cuda.get_device_capability()[0]
        if sm_version < 8:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHMLA"
            config['dtype'] = 'float32'
            os.environ["VLLM_USE_STANDALONE_COMPILE"] = "0"
            print(f"DEBUG: Setting VLLM_BACKEND=flashmla for sm_version {sm_version}")
        else:
            print(f"DEBUG: Using default VLLM_BACKEND for sm_version {sm_version}")

    if config['dtype'] == 'auto':
        if not torch.cuda.is_available():
            config['dtype'] = 'float32'

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = build_vllm_cli_args(config)
    neuron = os.environ.get("VLLM_TARGET_DEVICE") == "neuron"
    if neuron:
        os.environ.pop("NKI_COMPILER_FRONTEND", None)
        drop = ("--enable-lora", "--enable-tokenformer", "--gpu-memory-utilization",
                "--max-model-len", "--enable-auto-tool-choice", "--tool-call-parser",
                "--dtype", "--limit-mm-per-prompt")
        args = [a for a in args if a.split("=")[0] not in drop]

    extra_args = os.environ.get("SCALARLM_VLLM_ARGS", "")

    if extra_args:
        extra_args_list = shlex.split(extra_args) if neuron else extra_args.split()
        if neuron:
            sys.argv += [a for a in extra_args_list if a.startswith("--optimization-level")]

        for extra_arg in extra_args_list:
            arg_name = extra_arg.split("=")[0]
            args = [arg for arg in args if not arg.startswith(arg_name + "=")]

        args.extend(extra_args_list)
        print(f"DEBUG: Added extra args from SCALARLM_VLLM_ARGS: {extra_args_list}")

    print(f"DEBUG: About to parse args: {args}")
    print(f"DEBUG: Environment variables:")
    print(f"  SCALARLM_VLLM_ARGS: {os.environ.get('SCALARLM_VLLM_ARGS', 'NOT SET')}")
    print(f"  VLLM_TARGET_DEVICE: {os.environ.get('VLLM_TARGET_DEVICE', 'NOT SET')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

    args = parser.parse_args(args=args)

    args.port = port
    args.model = config["model"]

    logger.info(f"Running vLLM with args: {args}")

    await run_server(server_status, args)

async def run_server(server_status, args, **uvicorn_kwargs) -> None:
    decorate_logs("APIServer")

    if "reuse_port" in inspect.signature(setup_server).parameters:
        listen_address, sock = setup_server(args, reuse_port=False)
    else:
        listen_address, sock = setup_server(args)
    await run_server_worker(server_status, listen_address, sock, args, **uvicorn_kwargs)

async def run_server_worker(server_status, listen_address,
                            sock,
                            args,
                            client_config=None,
                            **uvicorn_kwargs) -> None:
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs['log_config'] = log_config

    async with build_async_engine_client(
            args,
            client_config=client_config,
    ) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        logger.info("Supported tasks: %s", supported_tasks)
        app = build_app(args, supported_tasks, model_config)

        server_status.set_app(app)

        await init_app_state(engine_client, app.state, args, supported_tasks)

        logger.info("Starting vLLM API server %d on %s", server_index,
                    listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    try:
        await shutdown_task
    finally:
        sock.close()
