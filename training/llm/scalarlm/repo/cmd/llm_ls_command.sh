inspect_args

./cray build-image

declare -a ls_command_parts
ls_command_parts=(
      "python" "/app/cray/sdk/masint/cli/main.py" "ls"
)

ls_command="${ls_command_parts[*]}"

echo $command

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ROOT_DIRECTORY=$LOCAL_DIRECTORY/..

declare -a docker_command_parts

mkdir -p $ROOT_DIRECTORY/data

docker_command_parts=("docker" "run" "--rm" "--network" "host")

docker_command_parts+=("cray:latest" "sh" "-c" "'$ls_command'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command

