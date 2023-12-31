# RL-traffic-control
Final Project repository for ELENE6885 - Reinforcement Learning </br>
Hongzhi Shi (hs3194), Karen Jan (kj2546), Lance Norman (ln2461), Siddarth Giddu (sg4170)

## IntelliLight
The modifications to IntelliLight can be found on the master branch in the DQN Folder. </br>

## FRAP
FRAP-t can be found in FRAP folder in the [transformer_dqn_agent.py](https://github.com/lancebnorman/RL-traffic-control/blob/main/FRAP/transformer_dqn_agent.py) file. </br>
FRAP-t2 can be found in the smart_transformer_frap_branch in the FRAP/modified_transformer.py </br>
FRAP-t2 was coded to be integrated and ran using the LibSignal simulator. The actual new agent can be found in [agent/frap_transformer.py](https://github.com/lancebnorman/RL-traffic-control/blob/main/LibSignal/agent/frap_transformer.py). As we had to create our own transformer and not use the one available in the pytorch library, the code for our new transformer can be found in [agent/modified_transformers.py](https://github.com/lancebnorman/RL-traffic-control/blob/main/LibSignal/agent/modified_transformers.py). The original LibSignal code used a DQN for all deep learning based RL agents. This meant that the base code didn't pass timestep to the Agents and used Relay Memories. This would not work for FRAP-t2 as we needed timestep for our positional encoding. Thus, we made changes to [trainer/tsc_trainer.py](https://github.com/lancebnorman/RL-traffic-control/blob/main/LibSignal/trainer/tsc_trainer.py). To see the ipynb used for experiments, go to [k_try2.ipynb](https://github.com/lancebnorman/RL-traffic-control/blob/main/k_try2.ipynb)
