# Trio Enterprise — Camera Source Registry

## Active Test Sources

### Scenario 1: Security / Building Monitoring (AIDC)

| Camera | Source | Type | Detections | Notes |
|--------|--------|------|-----------|-------|
| Office Camera | `rtsp://admin:***@192.168.1.100:554/h264Preview_01_sub` | RTSP | 5 cars, trucks | Reolink, residential driveway |
| Parking Garage | `https://youtube.com/watch?v=zBBVnq20HFU` | YouTube | 4 cars, 1 person | Indoor garage, vehicle tracking |
| India Crossroad | `https://youtube.com/watch?v=7aSkJCUDAes` | YouTube | 11 cars, 1 bus | High-density traffic intersection |
| Amsterdam Crossing | `https://youtube.com/watch?v=JvtvX7oXj70` | YouTube | 3 people, 2 bikes, 1 car | Bicycle + pedestrian, European urban |

### Scenario 2: Retail / Store / People Counting (Hedge Fund)

| Camera | Source | Type | Detections | Notes |
|--------|--------|------|-----------|-------|
| NYC 42nd Street | `https://youtube.com/watch?v=8uHdbMOqSwY` | YouTube | 7 people, 2 cars | **Best overall** — busy NYC street, mixed traffic |
| Cafe Entrance | `https://youtube.com/watch?v=sUOTOHCVGGk` | YouTube | 4 people | Barista cafe, customer counting |
| Busy Sidewalk | `https://youtube.com/watch?v=k4UXMsm4xqE` | YouTube | 1 person, 1 car, 1 bus | City sidewalk, moderate traffic |
| Hollywood Walk | `https://youtube.com/watch?v=9hz7BC-5Wds` | YouTube | 2 cars | LA street, better during daytime |

### Pending (Best during daytime US hours)

| Camera | Source | Type | Expected |
|--------|--------|------|---------|
| Venice Beach | `https://youtube.com/watch?v=9lFuecfQqiI` | YouTube | High pedestrian traffic (daytime) |
| OC Boardwalk | `https://youtube.com/watch?v=iuVnIe86-Zk` | YouTube | Beach + boardwalk foot traffic |
| OC Flamingo | `https://youtube.com/watch?v=p1pn9ANuntM` | YouTube | Beach boardwalk |

## Camera IDs in Database

| ID | Name | Source Type |
|----|------|------------|
| cam_c8603989 | Office Camera | RTSP |
| cam_6b97e346 | NYC 42nd Street | YouTube |
| cam_89d128ec | Amsterdam Crossing | YouTube |
| cam_ee0bf3d6 | India Crossroad | YouTube |
| cam_9bb93cc0 | Parking Garage | YouTube |
| cam_204141ff | Cafe Entrance | YouTube |

## Data Collected (as of 2026-03-21)

- **223 events** (102 with VLM descriptions, 30 crop-described)
- **673 metrics** (people_in/out, count_car, count_bus, count_person, etc.)
- **6 cameras** active

## Source Types Supported

| Type | Input Format | Backend Handler |
|------|-------------|----------------|
| RTSP | `rtsp://user:pass@host:port/path` | ffmpeg direct |
| YouTube | `https://youtube.com/watch?v=...` | yt-dlp → ffmpeg |
| HLS | `https://.../*.m3u8` | ffmpeg direct |
| HTTP | `http://...` | cv2.VideoCapture |
| Webcam | `0`, `1` | cv2.VideoCapture(int) |
| File | `/path/to/video.mp4` | cv2.VideoCapture |

## YOLO Detection Classes

| Class | COCO ID | Use Case |
|-------|---------|----------|
| person | 0 | Both — foot traffic + security |
| bicycle | 1 | Urban traffic |
| car | 2 | Parking, traffic |
| motorcycle | 3 | Traffic |
| bus | 5 | Public transit tracking |
| truck | 7 | Logistics, loading dock |
| dog | 16 | Misc detection |
| cat | 17 | Misc detection |

## VLM Crop-Describe Prompt Quality (Autoresearch Results)

| Prompt Version | Score | Latency | Best For |
|---------------|-------|---------|----------|
| v2_structured | 99.3/100 | 670ms | Production — balanced |
| v4_concise | 89.7/100 | 476ms | High-throughput |
| v3_detailed | 100/100 | 1447ms | Deep analysis |

### Sample Crop-Describe Outputs
```
[person #24] Elderly woman, Asian ethnicity, long coat, carrying bag, walking
[person #29] Middle-aged man, dark suit, carrying briefcase, walking
[car #80] Silver Toyota Corolla, compact sedan, distinctive front grille
[bus #669] Red and white bus from 1950s, mid-20th century model
[person #74] Elderly man, gray hair, long-sleeved shirt, carrying walking stick
```
