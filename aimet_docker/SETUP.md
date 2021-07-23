# Execute the following commands on terminal
1. Setup the environment

```WORKSPACE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"```
2. Export the python path

```export PYTHONPATH=$WORKSPACE/build/staging/universal/lib/x86_64-linux-gnu:$WORKSPACE/build/staging/universal/lib/python:$PYTHONPATH```

3. Export the library path

```export LD_LIBRARY_PATH=$WORKSPACE/build/staging/universal/lib/x86_64-linux-gnu:$WORKSPACE/build/staging/universal/lib/python:$LD_LIBRARY_PATH```
