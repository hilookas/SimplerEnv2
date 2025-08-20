# Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation

<div align="center">

**Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation**

[[🌐 Website](https://embodied-r1.github.io)] [[📄 Paper](#paper)] [[🤗 Models](https://huggingface.co/collections/IffYuan/embodied-r1-684a8474b3a49210995f9081)] [[🎯 Datasets](https://huggingface.co/collections/IffYuan/embodied-r1-684a8474b3a49210995f9081)] [[💬 Demo](#demo)]

</div>

---

This repository is used to evaluate the ER1 performance on bridge tasks of [SimplerEnv](https://github.com/simpler-env/SimplerEnv). 


## 💿 Installation

**Clone this repo:**

```bash
git clone --recurse-submodules https://github.com/hilookas/SimplerEnv2 SimplerEnv -b er1
cd SimplerEnv
```

**Create an anaconda environment:**

```bash
conda create -n simpler_env python=3.10
conda activate simpler_env
```

**Install SimplerEnv:**

```bash
# Following the instructions <https://github.com/simpler-env/SimplerEnv#installation>

pip install numpy==1.24.4

pushd ManiSkill2_real2sim
pip install -e .
popd

pip install -e .
```

**Install GraspNet:**

```bash
# Following the instructions in GSNet/README.md

pushd GSNet

pushd pointnet2
python setup.py install
popd

pushd knn
python setup.py install
popd

popd
```

## 🏃 Execution

You can run the evaluation using:

```bash
bash scripts/er1_bridge.sh
```

## 🙏 Acknowledgments

We sincerely thank the following open-source projects and research works:

- [SimplerEnv-SOFAR](https://github.com/Zhangwenyao1/SimplerEnv-SOFAR)
- [ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim)
- [graspnetAPI](https://github.com/graspnet/graspnetAPI)
- [graspness_unofficial](https://github.com/graspnet/graspness_unofficial) / [graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d)

## 📜 Citation

If you use our work in your research, please cite our paper:

```bibtex
@article{yuan2025embodiedr1,
  title={Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation},
  author={Yuan, Yifu and Cui, Haiqin and Huang, Yaoting and Chen, Yibin and Ni, Fei and Dong, Zibin and Li, Pengyi and Zheng, Yan and Hao, Jianye},
  year={2025}
}
```
