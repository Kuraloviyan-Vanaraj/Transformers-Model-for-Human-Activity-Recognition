# Dataset setup

The dataset is **not included in this repository**.

This project uses the inertial component of the **UTD-MHAD (UTD Multimodal Human Action Dataset)**. Download `Inertial_Data.zip` from the official UTD-MHAD page maintained by the University of Texas at Dallas:

https://personal.utdallas.edu/~kehtar/UTD-MHAD.html

After downloading, extract the inertial `.mat` files into:

```text
data/raw/inertial/
```

The notebooks expect filenames such as:

```text
a1_s1_t1_inertial.mat
a27_s8_t4_inertial.mat
```

The official dataset page describes UTD-MHAD as 27 actions performed by 8 subjects, with 861 sequences after three corrupted sequences were removed. It also documents the original dataset authors and the required citation.

## Dataset citation

Chen, C., Jafari, R., & Kehtarnavaz, N. (2015). *UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor*. Proceedings of the IEEE International Conference on Image Processing (ICIP).

Please follow the dataset creators' terms and citation requirements when using the data.
