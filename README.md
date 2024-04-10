# Multi-alignment-Drivng-Agent

This repository contains the multi-alignment framework code for paper **Driving Style Alignment for LLM-powered Driver Agent**.

## Installation and Configuration Guide
Before running our code, the installation and configuration of CARLA is required, for which we recommend version 0.9.14, the same as used in the paper. We also recommend using Python 3.7 and Unreal Engine 4.

The detailed build guide can be referred to in [CARLA Simulator](https://carla.readthedocs.io/en/latest/). In the bottom right corner, select the corresponding CARLA version.

After successfully running CARLA, follow these steps to intergrate our code:

1. Replace the `automatic_control.py` in the CARLA source code at `carla/PythonAPI/examples/automatic_control.py` with `Multi-alignment-Driving-Agent/src/automatic_control.py`
2. Replace the `navigation` folder in the CARLA source code at `/carla/PythonAPI/carla/agents/navigation` with `Multi-alignment-Driving-Agent/src/navigation`
3. In `navigation/behavior_agent.py`, configure your API keys and organization of OpenAI, as well as the path of prompts and guidelines.

## Running

Run the following command in the `carla/PythonAPI/examples/` directory:

```shell
conda activate carla
python automatic_control.py
```

