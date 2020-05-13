# Multi-Subject Tracking

Source code of my Master's Thesis -- Multi-Subject Tracking in Crowded Videos.

> The affordability of low-cost cameras implies vast amounts of multimedia recordings, whose automated processing is a challenging task. One of the interesting processing paradigms that extract valuable knowledge from videos is multi-subject tracking in crowded videos. Multi-subject tracking aims to detect subjects across consecutive video frames as well as to maintain their identities in time. Moreover, it has great application potential in topics like crowd behavior analysis, detection of organized crime, and pedestrian tracking by self-driving vehicles. These applications require both effective and efficient solutions with real-time response. Many state-of-the-art trackers utilizing visual features work effectively but not efficiently due to the usage of complex computational tools like neural networks. In this thesis, we propose a high-speed online tracking algorithm that utilizes either bounding box or skeleton data. Furthermore, we introduce several enhancements (e.g., performance boost with the usage of sub-optimal greedy algorithm, naive re-identification, interpolation of missing detections) that one-by-one as well as altogether improve the tracking performance and accuracy. In order to demonstrate usability, we evaluate the proposed method on densely populated videos (as much as 40 subjects per frame) from the standardized dataset. We reach competitive results in accuracy and outperform the majority of state-of-the-art trackers in tracking speed.

## Quick start

### Clone

Clone this repo to your local machine using `https://github.com/matusmacko/multi-subject-tracking`

### Installation

Install dependencies
```
pip install -r requirements.txt
```

### Dataset 
Download datasets from [MOT Challenge website](https://motchallenge.net).

### Configuration

Set the tracker configuration in [settings](src/settings/__init__.py).

### Run
Run tracker using  `inv process`.


### Evaluation

Use publicly available [py-motmetrics](https://github.com/cheind/py-motmetrics) library.

## Built With

* [Munkres](http://software.clapper.org/munkres/) - Python implementation of Munkres algorithm
* [Invoke](https://github.com/pyinvoke/invoke) - Pythonic task management & command execution


## Authors

* **Matúš Macko**
