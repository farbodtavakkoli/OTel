set -Eeuoxa pipefail

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

TTY=-t
if test -t 0; then
  TTY=-it
fi

docker run --rm $TTY --user $(id -u):$(id -g) \
    --volume "$LOCAL_DIRECTORY:/app/cmd" \
    --volume "$LOCAL_DIRECTORY/../scripts:/app/scripts" \
    --volume "$LOCAL_DIRECTORY/bashly-settings.yml:/app/bashly-settings.yml" \
    dannyben/bashly "$@"
