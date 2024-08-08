---
permalink: /
title: "Lei Fan's Homepage"
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

**Greetings from Evanston.** 

I am currently a Ph.D. candidate at Northwestern University under the supervision of [Prof. Ying Wu](http://users.ece.northwestern.edu/~yingwu/). My research interests lies in **the intersection of computer vision and robotics**, with a particular emphasis on <ins>**active vision**</ins> (<ins>*the agent is endowed with the ability to move and perceive*</ins>).
I am constantly investigating the challenges inherent to active vision agents in an open-world context. These challenges include, but are not limited to, *continual learning*, *few-sample learning*, *uncertainty quantification* and *vision-language models*.

Prior to my Ph.D., my researches primarily focused on the perception in autonomous driving vehicles, encompassing areas such as stereo vision, 3D mapping, moving-object detection and map repair. 

My detailed resume/CV is [here](./files/Lei_Fan_Resume.pdf) (last updated on July 2024).

# 🔥 News
- *2024.05*: The proposed dataset to evaluate active recognition has been made publicly available! Please refer to [the page](AR-dataset/index.html) for details.
- *2024.04*: &nbsp;🎉 I have successfully defended my Ph.D.! I would like to extend my gratitude to my committee: Prof. Ying Wu, Prof. Qi Zhu, and Prof. Thrasos N. Pappas. And I will join [Amazon Robotics](https://www.amazon.science/research-areas/robotics) as an Applied Scientist this summer!
- *2024.02*: &nbsp;🎉 Two papers on *active recognition* for embodied agents have been accepted by CVPR 2024! Thanks to all my collaborators!
- *2023.07*: &nbsp;🎉 Our paper on *uncertainty estimation* has been accepted to ICCV 2023! Appreciation goes out to all advisors: Dr. Bo Liu, Dr. Haoxiang Li, Prof. Ying Wu, and Prof. Gang Hua!

# 📖 Educations
- *2019 - 2024*, M.S., Ph.D. in Electrical Engineering, advised by [Prof. Ying Wu](http://users.ece.northwestern.edu/~yingwu/), Northwestern University.
- *2013 - 2019*, B.E., M.S. in Computer Science, advised by [Prof. Long Chen](https://scholar.google.com/citations?user=jzvXnkcAAAAJ&hl=zh-CN), Sun Yat-sen University.

# 📝 Publications

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2024</div><img src='images/overview/AOVR.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
[Active Open-Vocabulary Recognition: Let Intelligent Moving Mitigate CLIP Limitations](https://arxiv.org/pdf/2311.17938.pdf)

**Lei Fan**, Jianxiong Zhou, Xiaoying Xing, Ying Wu

[**Project (coming soon)**]() <strong><span class='' data=''></span></strong> |
[**Video**]() <strong><span class='' data=''></span></strong>
- Investigate CLIP's limitations in embodied perception scenarios, emphasizing diverse viewpoints and occlusion degrees.
- Propose an active agent to mitigate CLIP's limitations, aiming for active open-vocabulary recognition.

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2024</div><img src='images/overview/EAR.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
[Evidential Active Recognition: Intelligent and Prudent Open-World Embodied Perception](https://arxiv.org/pdf/2311.13793.pdf)

**Lei Fan**, Mingfu Liang, Yunxuan Li, Gang Hua, Ying Wu

[**Supplementary**](./files/CVPR_2024_EAR_supplemenraty.pdf) <strong><span class='' data=''></span></strong> | 
[**Dataset**](AR-dataset/index.html) <strong><span class='' data=''></span></strong> | 
[**Project (coming soon)**]() <strong><span class='' data=''></span></strong> |
[**Video**]() <strong><span class='' data=''></span></strong>
- Handling unexpected visual inputs for embodied agent's training and testing in open environments.
- Collect a dataset for evaluating active recognition agents. Each testing sample is accompanied with a recognition difficulty level.
- Applying evidential deep learning and evidence combination for frame-wise information fusion, mitigating unexpected image interference.

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICCV 2023</div><img src='images/overview/ICCV23.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
[Flexible Visual Recognition by Evidential Modeling of Confusion and Ignorance](https://arxiv.org/pdf/2309.07403.pdf)

**Lei Fan**, Bo Liu, Haoxiang Li, Ying Wu, Gang Hua

[**Supplementary**](./files/ICCV_2023_Supplementary.pdf) <strong><span class='' data=''></span></strong> | 
[**Poster**](./files/ICCV_2023_Poster.pdf) <strong><span class='' data=''></span></strong> |
[**Project**](flexible-recognition/index.html) <strong><span class='' data=''></span></strong>
- Modeling both confusion and ignorance with hyper-opinions.
- Proposing a hierarchical structure with binary plausible functions to handle the challenge of 2^K predictions.
- Experiments with synthetic data, flexible visual recognition, and open-set detection validate our approach.

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">WACV 2023</div><img src='images/overview/WACV23.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
[Avoiding Lingering in Learning Active Recognition by Adversarial Disturbance](https://openaccess.thecvf.com/content/WACV2023/papers/Fan_Avoiding_Lingering_in_Learning_Active_Recognition_by_Adversarial_Disturbance_WACV_2023_paper.pdf)

**Lei Fan**, Ying Wu

[**Supplementary**](./files/WACV_2023_Supplementary.pdf) <strong><span class='' data=''></span></strong> | 
[**Poster**](./files/WACV_2023_Poster.pdf) <strong><span class='' data=''></span></strong>
- *Lingering*: The joint learning process could lead to unintended solutions, like a collapsed policy that only visits views that the recognizer is already sufficiently trained to obtain rewards.
- Our approach integrates another adversarial policy to disturb the recognition agent during training, forming a competing game to promote active explorations and avoid lingering.
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICCV 2021</div><img src='images/overview/ICCV21.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
[FLAR: A Unified Prototype Framework for Few-sample Lifelong Active Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_FLAR_A_Unified_Prototype_Framework_for_Few-Sample_Lifelong_Active_Recognition_ICCV_2021_paper.pdf)

**Lei Fan**, Peixi Xiong, Wei Wei, Ying Wu

[**Supplementary**](./files/ICCV_2021_Supplementary.pdf) <strong><span class='' data=''></span></strong> | 
[**Poster**](./files/ICCV_2021_Poster.pdf) <strong><span class='' data=''></span></strong>
- The active recognition agent needs to incrementally learn new classes with limited data during exploration.
- Our approach integrates prototypes, a robust representation for limited training samples, into a reinforcement learning solution, which motivates the agent to move towards views resulting in more discriminative features.
</div>
</div>

<!-- - [Avoiding Lingering in Learning Active Recognition by Adversarial Disturbance](https://openaccess.thecvf.com/content/WACV2023/papers/Fan_Avoiding_Lingering_in_Learning_Active_Recognition_by_Adversarial_Disturbance_WACV_2023_paper.pdf), **Lei Fan**, Ying Wu, accepted by IEEE/CVF Winter Conference on Applications of Computer Vision (**WACV**), 2023. -->

- [Unsupervised Depth Completion and Denoising for RGB-D Sensors](https://ieeexplore.ieee.org/document/9812392), **Lei Fan**, Yunxuan Li, Chen Jiang, Ying Wu, accepted by IEEE International Conference on Robotics and Automation (**ICRA**), 2022.

<!-- - [FLAR: A Unified Prototype Framework for Few-sample Lifelong Active Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_FLAR_A_Unified_Prototype_Framework_for_Few-Sample_Lifelong_Active_Recognition_ICCV_2021_paper.pdf), **Lei Fan**, Peixi Xiong, Wei Wei, Ying Wu, accepted by IEEE International Conference on Computer Vision (**ICCV**), 2021. -->

- [SemaSuperpixel: A Multi-channel Probability-driven Superpixel Segmentation Method](https://ieeexplore.ieee.org/document/9506437), Xuehui Wang, Qingyun Zhao, **Lei Fan**, Yuzhi Zhao, Tiantian Wang, Qiong Yan, Long Chen, accepted by IEEE International Conference on Image Processing (**ICIP**), 2021.

- [Lightweight Single-Image Super-Resolution Network with Attentive Auxiliary Feature Learning](https://arxiv.org/pdf/2011.06773.pdf), Xuehui Wang, Qing Wang, Yuzhi Zhao, Junchi Yan, **Lei Fan**, and Long Chen, accepted by Asian Conference on Computer Vision (**ACCV**), 2020.

- [Toward the Ghosting Phenomenon in a Stereo-Based Map With a Collaborative RGB-D Repair](https://ieeexplore.ieee.org/abstract/document/8701620), Jiasong Zhu, **Lei Fan**, Wei Tian, Long Chen, Dongpu Cao, and Fei-Yue Wang, accepted by IEEE Transactions on Intelligent Transportation Systems (**Tran-ITS**), 2020.

- [Monocular Outdoor Semantic Mapping with a Multi-task Network](https://arxiv.org/pdf/1901.05807.pdf), Yucai Bai, **Lei Fan**, Ziyu Pan, and Long Chen, accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (**IROS**), 2019.

- [Collaborative 3D Completion of Color and Depth in a Specified Area with Superpixels](https://www.researchgate.net/publication/328156601_Collaborative_3D_Completion_of_Color_and_Depth_in_a_Specified_Area_with_Superpixels), **Lei Fan**, Long Chen, Chaoqiang Zhang, Wei Tian, and Dongpu Cao, accepted by IEEE Transactions on Industrial Electronics (**TIE**), 2018.

- [Planecell: Representing Structural Space with Plane Elements](https://ieeexplore.ieee.org/document/8500416), **Lei Fan**, Long Chen, Kai Huang and Dongpu Cao, accepted as Best Student Paper by IEEE Intelligent Vehicles Symposium (**IV**), 2018.

- [A Full Density Stereo Matching System Based on the Combination of CNNs and Slanted-planes](https://ieeexplore.ieee.org/document/8103909), Long Chen, **Lei Fan**, Jianda Chen, Dongpu Cao, and Feiyue Wang, accepted by IEEE Transactions on Systems, Man, and Cybernetics: Systems (**TSMCS**), 2017.

- [Let the Robot Tell: Describe Car Image with Natural Language via LSTM](https://www.sciencedirect.com/science/article/pii/S016786551730315X), Long Chen, Yuhang He, and **Lei Fan**, accepted by Pattern Recognition Letters (**PRL**), 2017.

- [Moving-Object Detection from Consecutive Stereo Pairs using Slanted Plane Smoothing](https://ieeexplore.ieee.org/document/7891876), Long Chen, **Lei Fan**, Guodong Xie, Kai Huang, and Andreas Nuchter, accepted by IEEE Transactions on Intelligent Transportation Systems (**Tran-ITS**), 2017.

- [RGB-T SLAM: A Flexible SLAM Framework by Combining Appearance and Thermal Information](https://ieeexplore.ieee.org/abstract/document/7989668), Long Chen, Libo Sun, Teng Yang, **Lei Fan**, Kai Huang, and Zhe Xuanyuan, accepted by IEEE International Conference on Robotics and Automation (**ICRA**), 2017.


<!-- # 💬 Invited Talks
- *2021.06*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2021.03*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.  \| [\[video\]](https://github.com/) -->

# 💻 Internships
- *2023.06 - 2023.09*, Applied Scientist Intern, [Amazon Robotics](https://www.amazon.science/research-areas/robotics), Seattle, US.<br />
  <span style="color:grey">- Topic: Surface normal estimation and stability analysis.</span><br />
  <span style="color:grey">- Advisors: Dr. Shantanu Thaker, Dr. Sisir Karumanchi.</span>
- *2022.06 - 2022.09*, Research Intern, [Wormpex AI Research](http://research.wormpex.com/), Bellevue, US.<br />
  <span style="color:grey">- Topic: Uncertainty quantification for deep visual recognition.</span><br />
  <span style="color:grey">- Advisors: Dr. Bo Liu, Dr. Haoxiang Li, and Dr. Gang Hua.</span>
- *2020.06 - 2020.09*, Research Intern, Yosion Analytics, Chicago, US.<br />
  <span style="color:grey">- Topic: Autonomous forklift in a human-machine co-working environment.</span>
- *2016.06 - 2016.09*, Visual Engineer Intern, [DJI](https://www.dji.com/), Shenzhen, China.<br />
  <span style="color:grey">- Topic: Stereo matching using the fish-eye cameras on drones.</span>

# 🎖 Honors and Awards
- *2019.09* Northwestern University Murphy Fellowship.
- *2018.06* Best Student Paper, IEEE Intelligent Vehicle Symposium.
- *2019.09* National Merit Scholarship, China
