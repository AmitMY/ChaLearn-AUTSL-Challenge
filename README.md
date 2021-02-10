# AUTSL SLR

## Setup Weights and Biases

- Go to [wandb](https://wandb.ai/)
- Sign up, setup a project named `autsl`
- Run from terminal: `wandb login`

## Experiments to try?
### Taken by Amit:
- Amit: Try using angle + distance representations
- Amit: Look into the horizontal flip, if its performing correctly
- Amit: Add temporal augmentation

### Not taken
- Reduce data augmentation by a lot, to see the effects
- Experiment with using the face
- Add Z and K dimensions (POSE_DIMS = 4) rather than just X and Y
- Run pose estimation on flipped videos to increase data
- Change the Linformer to a basic transformer, or LSTM
- Change the scale of the positional encoding
- Change the scale of the input to the transformer (mult by 10, div by 10)
