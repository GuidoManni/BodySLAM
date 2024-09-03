# BodySLAM: A Generalized Monocular Visual SLAM Framework for Surgical Applications

[![arXiv](https://img.shields.io/badge/arXiv-2408.03078-b31b1b.svg)](https://arxiv.org/abs/2408.03078)

BodySLAM is a cutting-edge, deep learning-based Simultaneous Localization and Mapping (SLAM) framework designed specifically for endoscopic surgical applications. By leveraging advanced AI techniques, BodySLAM brings enhanced depth perception and 3D reconstruction capabilities to various surgical settings, including laparoscopy, gastroscopy, and colonoscopy.

## 📄 Research Paper

Our comprehensive paper detailing the BodySLAM framework is now available on arXiv:

**[BodySLAM: A Generalized Monocular Visual SLAM Framework for Surgical Applications](https://arxiv.org/abs/2408.03078)**

*G. Manni, C. Lauretti, F. Prata, R. Papalia, L. Zollo, P. Soda*

If you find our work useful in your research, please consider citing:

```bibtex
@misc{manni2024bodyslamgeneralizedmonocularvisual,
      title={BodySLAM: A Generalized Monocular Visual SLAM Framework for Surgical Applications}, 
      author={G. Manni and C. Lauretti and F. Prata and R. Papalia and L. Zollo and P. Soda},
      year={2024},
      eprint={2408.03078},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.03078}, 
}
```

## 🚀 Overview

In the challenging world of endoscopic surgeries, where hardware limitations and environmental variations pose significant obstacles, BodySLAM stands out by integrating deep learning models with strong generalization capabilities. Our framework consists of three key modules:

1. **Monocular Pose Estimation Module (MPEM)**
2. **Monocular Depth Estimation Module (MDEM)**
3. **3D Reconstruction Module (3DM)**

## ✨ Features

- **State-of-the-Art Depth Estimation**: Utilizes the Zoe model for accurate monocular depth estimation
- **Novel Pose Estimation**: Implements CyclePose, an unsupervised method based on CycleGAN architecture
- **Cross-Setting Performance**: Robust functionality across various endoscopic surgical environments

## 🛠 Refactoring Status

We're actively refactoring our codebase to enhance usability and performance. Here's our current progress:

- [x] Monocular Depth Estimation Module (MDEM)
- [ ] Monocular Pose Estimation Module (MPEM)
- [ ] 3D Reconstruction Module (3DM)
- [ ] Integration and Testing

## 📘 Examples

We've included several examples to help you get started with BodySLAM, particularly with the Depth Estimation module:

1. **Basic Depth Estimation**: Demonstrates the fundamental pipeline for estimating depth from a single image.
   - File: `examples/depth_estimation/basic_depth_estimation.py`

2. **Depth Map Scaling and Colorization**: Shows how to scale and colorize depth maps for better visualization.
   - File: `examples/depth_estimation/depth_map_scaling.py`

3. **Batch Processing**: Illustrates how to process multiple images for depth estimation and colorization.
   - File: `examples/depth_estimation/batch_processing.py`

To run these examples:
1. Navigate to the `examples/depth_estimation/` directory
2. Run the desired script, e.g., `python basic_depth_estimation.py`

## 🔜 Coming Soon

- **Enhanced Documentation**: We're working on detailed installation instructions, usage guidelines, and more examples.


## 🚀 Installation

(Detailed installation instructions will be provided upon completion of code refactoring.)

## 🔧 Usage

(Comprehensive usage guidelines will be added after refactoring and documentation improvements.)

## 🤝 Contributing

We welcome contributions! If you're interested in improving BodySLAM, please check our [Contributing Guidelines](CONTRIBUTING.md) (coming soon).

## 📄 License

BodySLAM is released under the [MIT License](LICENSE).

---

For questions or support, please [open an issue](https://github.com/yourusername/BodySLAM/issues) on our GitHub repository.
