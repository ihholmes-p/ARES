
# ARES: Attention-based REward Shaping

ARES uses reward shaping to help solve sparse and delayed reward environments.

## Requirements

- **Python version** (can be different, but this was used in the experiments): 3.10.12  
- **Dependencies**:  
  Install via pip:
  ```bash
  pip install -r requirements.txt
  ```

  Note: To use MuJoCo on Windows, we recommend the usage of [WSL](https://learn.microsoft.com/en-us/windows/wsl/install). 

## Datasets

The datasets (TrainingExpert and Random) used in the paper can be downloaded [here](https://drive.google.com/file/d/13ue2aqBMZLj_6IxhZYW4Azz8id5dNN-G/view?usp=sharing).

The MuJoCo environment shaped rewards can be downloaded [here](https://drive.google.com/file/d/1mtUXWjigaX4I1etbTfqnTfF7jglbVE7o/view?usp=sharing).

---

## Running MuJoCo Experiments

### Generate Random Data

Generate 10 random datasets of 1000 timesteps each using 4 processes:

```bash
python gym_random_episode_gen.py Hopper-v4 1000 4
```

### Generate TrainingExpert Data

Generate 1,000,000 timesteps of a SAC agent learning `Hopper-v4`:

```bash
python generate_SAC_expert_output.py output.txt Hopper-v4 1000000 4
```

- Output will be saved to `output.txt`

### Generate Shaped Rewards from Output

Train a transformer reward model for 10,000 epochs on the generated output:

```bash
python output_to_unmerged_rewards.py output.txt rewards.txt 10000 512 0.00001 8 11 3
```

- `512`: embedding size  
- `0.00001`: learning rate  
- `8`: internal (final-layer) embedding size  
- `11`: state dimensions  
- `3`: action dimensions (Hopper-v4)

### Train SAC Agent with Shaped Rewards

Train a SAC agent using the shaped rewards:

```bash
python generate_SAC_inferred_output.py rewards.txt finaloutput.txt Hopper-v4 4 1000000 11 3 5 1 1 0
```

- `4`: number of vectorized environments  
- `1000000`: timesteps  
- `11/3`: state/action dimensions  
- `5 1 1 0`: distance hyperparameters  
- Output saved to `finaloutput.txt`

---

## Toy Environments (CliffWalking-m, CartPole, LunarLander)

To generate data, run the respective generator files under each folder.

To rerun our experiments, move the data under the same directory as the relevant experiment generator Python file, then run the experiment generator.

## Notes

---

There are 3 main Python files for SAC on MuJoCo (one for shaped rewards, one for immediate rewards, and one for shaped rewards).

Similarly, there are 3 main Python files for PPO on MuJoCo.

There is one more Python file for SAC on MuJoCo using the shaped rewards: `generate_SAC_inferred_output_parameters.py` which can be used for a hyperparameter search using the shaped rewards. It takes 4 extra parameters, which we only evaluated on the HalfCheetah environment in an attempt to improve performance (for example, one is the SAC learning rate, which as noted in Appendix B is best set at 1e-5 when using our shaped rewards). For these 4 parameters, the default values are used for the other environments. This information is under Appendix B in the paper.

Note that all of these Python files share a good deal of common code: the main reason it is set up this way is to make it very, very clear (both for us while performing the research, and for the interested reader) that there is absolutely no leakage between the three controlled experiment settings (i.e. the immediate rewards are never, ever used by the shaped reward agent). 
