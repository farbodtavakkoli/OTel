inspect_args

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ROOT_DIRECTORY=$LOCAL_DIRECTORY/../sdk

cd $ROOT_DIRECTORY

wheel_build_command="python -m build --sdist --wheel --outdir dist/ ."

echo $(green_bold Building wheel with command: ${wheel_build_command})
eval $wheel_build_command

echo $(green_bold Successfully built wheel)

pypi_upload_command="twine upload dist/*"

echo $(green_bold Uploading wheel to pypi with command: ${pypi_upload_command})
eval $pypi_upload_command

echo $(green_bold Successfully uploaded wheel to pypi)

