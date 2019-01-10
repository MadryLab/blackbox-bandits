# Prior Convictions: Black-box Adversarial Attacks with Bandits and Priors
This is the code for reproducing the paper "Prior Convictions: Black-box Adversarial Attacks with Bandits and Priors" ([arxiv](https://arxiv.org/abs/1807.07978)) to appear at ICLR 2019. The paper can be cited as follows:

```
@article{IEM2018PriorCB,
  title={Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors},
  author={Andrew Ilyas and Logan Engstrom and Aleksander Madry},
  journal={ICLR 2019},
  year={2018},
  url={https://arxiv.org/abs/1807.07978}
}
```

# Results
|                        | Avg Queries |          | Failure Rate |            | Avg Queries on NES success |         |
|------------------------|-------------|----------|--------------|------------|----------------------------|---------|
| Method                 |    l-inf    |    l-2   |     l-inf    |     l-2    |            l-inf           |   l-2   |
| NES                    |     1735    |   2938   |    22.2\%    |   34.4\%   |            1735            |   2938  |
| Bandits[T] (ours)      |     1781    |   2690   |    11.6\%    |   30.4\%   |            1214            |   2421  |
| __Bandits[TD]__ (ours) |   __1117__  | __1858__ |   __4.6\%__  | __15.5\%__ |           __703__          | __999__ |

# Reproducing the results

## Requirements
- Pytorch (`torch`, `torchvision`) packages
- `argparse` package

The results can be reproduced (with the default hyperparameters) with the following command:
```
python main.py [--nes] [--tiling] --json-config [configs/l2.json | configs/linf.json | configs/linf-nes.json | configs/l2-nes.json]
```

You can run ```python main.py --help``` to see all of the available options/hyperparameters. Although the hyperparameters were tuned for Inception-v3, the attack can by run with the flag `--classifier {inception_v3,resnet50,vgg16_bn}` to attack other classifiers.
