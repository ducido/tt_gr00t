



## General environment

```bash
git clone https://github.com/ducido/tt_gr00t

uv sync
uv pip install -e .
```

## Simpler Env
```
git clone https://github.com/squarefk/SimplerEnv.git external_dependencies/SimplerEnv
bash gr00t/eval/sim/robocasa/setup_SimplerEnv.sh
```

```bash
# first start a server
bash myscripts/server.sh wdx 5555
# then eval
bash myscript/base_wd.sh 5555
```

## Robocasa

```
git clone https://github.com/squarefk/robocasa external_dependencies/robocasa
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```

```bash
# first start a server
bash myscripts/server.sh robocasa 5550
# then eval
bash myscript/base_robocasa.sh 5550
```