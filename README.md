# VideosLabelHierarchy


## Datasets

We provide tools for downloading videos from youtube, cutting videos into clips, extracting frames from clips, and finally build the datasets readily for the `torch.Dataset`.

## 1. Download the Videos

Run the following command to download the videos from the youtube:

```bash
python -m tools.download.py [-h] -d {tennis,FineDiving,FineGym,fs_comp,fs_perf} [-f path/to/ffmpeg/executable] [-o path/to/save/the/videos] [-p port_of_proxy] [-i ip_of_proxy] [-n number_of_downloading_threads]
```

## 2. Create the Datasets

To cut videos into clip and extract frames from the clip, run the following commands:

```bash
python -m tools.make_datasets [-h] -d {tennis,FineDiving,FineGym,fs_comp,fs_perf} -i path/to/downloaded/videos [-o path/to/extracted/frames] [-m maximum_height_of the extracted frame] [-n number_of_extracting_frames]
```

Current support datasets:
- tennis
- FineGym
- FineDiving
- fs_comp (FineSkating)

## 3. Notice

Some of the videos contained in the datasets have been invalid, so we removed the invalid videos. Invalid videos are shown in

```bash
cat tools/{tennis,FineDiving,FineGym,fs_comp,fs_perf}/invalid-videos.csv
```

And also, annotations of the invalid videos are also removed. If you want to obtain the original annotations, check: https://github.com/jhong93/spot.git