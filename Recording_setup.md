# Cages

<p align="left">
    <img src=".//assets/assembled_cage.png" alt="Camera diagram" style="width: 400px; height: auto;">
</p>
<p align="left"> 

## Laser cut parts
| Part | Purpose | Needed Per Cage | Notes |
|:---| :--- |:---|:--- |
|[Cage_Blank_Wall.svg](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Cage_Blank_Wall.svg)| Side wall | 2 | White acrylic, 0.22" |
|[Cage_Water_Wall.svg](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Cage_Water_Wall.svg)| Water hopper wall | 1 | Clear acrylic, 0.22" |
|[Cage_Lid_No_Hole.svg](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Cage_Lid_No_Hole.svg) | Lid | 1 | Clear acrylic, 0.118" |
| [Cage_Food_Wall.svg](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Cage_Food_Wall.svg) | Food hopper wall | 1 | Clear acrylic, 0.22" |
| [Cage_Floor.svg](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Cage_Floor.svg)| Floor | 1 | White acrylic, 0.22" |

## Laser cut parts (optional)
| Part | Purpose | Needed Per Cage | Notes |
|:---| :--- |:---|:--- |
|[Cage_Wheel_Wall.svg](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Cage_Wheel_Wall.svg)| Side wall for running wheel | 1 | White acrylic, 0.22" |
|[Cage_Lid_Hole.svg](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Cage_Lid_Hole.svg)| Lid with hole for cable | 1 | Clear acrylic, 0.118" |

## 3D printed parts
| Part | Purpose | Needed Per Cage | Notes |
|:---| :--- |:---|:--- |
|[Mouse_Bottle_Holder.stl](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Mouse_Bottle_Holder.stl)| Water hopper | 1 | PLA, 0.2mm layer height, 15% infill |
|[Food_Hopper_No_Bars.stl](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Food_Hopper_No_Bars.stl)| Food hopper | 1 | PLA, 0.2mm layer height, 15% infill|

## 3D printed parts (optional)
| Part | Purpose | Needed Per Cage | Notes |
|:---| :--- |:---|:--- |
|[Mouse_Running_Wheel.3mf](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/Mouse_Running_Wheel.3mf)| Running wheel | 1 | PLA, 0.2mm layer height, 15% infill |

## Cage posts
| Part | Purpose | Needed Per Cage | Notes |
|:---| :--- |:---|:--- |
|T-slot framing (TNutz EX-1010 or similar)| Cage assembly | 4 | 8" length, 1"x1", M6 taps on each end|

## Assembly materials
| Part | Purpose | Needed Per Cage | Notes |
|:---| :--- |:---|:--- |
|Clear cast epoxy | Seals 3D printed parts |  | Alumilite 10590 or similar |
| Wire | Food hopper bars | | 16 gauge, galvanized |
| M6 screws | Cage assembly | 4 | M6-1.0x20mm |
|Water bottle nozzle| | 1 | Elbow, 75 mm|
|Water bottle||1|55 mm x 130 mm|

To epoxy the 3D printed parts:
1. Under a fume hood, mix a 1:1 ratio of resins A and B using disposable medicine cups and wooden tongue depressors.
2. Mixing is done when the solution turns transparent.
3. Coat the parts with a thin layer of epoxy and let dry in the hood overnight.

# Camera system

## IR illumination
| Part | Purpose | Needed | Notes |
|:---| :--- |:---|:--- |
|DC 1 female to 4 male output power splitter | IR power supply | 1 per 4 cages | [Amazon Link](https://www.amazon.com/3-Pack-Splitter-Adapter-Security-Cameras/dp/B07YZJZSLK)|
| Hookup wire | IR power supply | 3' per cage | [Amazon Link](https://www.amazon.com/TYUMEN-Electrical-Extension-Flexible-Lighting/dp/B07DDG7J9K) |
| Female 12V DC power jack adapter| IR power supply | 1 per cage| [Amazon Link](https://www.amazon.com/Ksmile%C2%AE-Female-2-1x5-5mm-Adapter-Connector/dp/B015OCV5Y8/r) |
| 850 nm IR light stripping| IR illumination | 1' per cage | [Amazon Link](https://www.amazon.com/SMD3528-300-IR-Infrared-Flexible-8mm-Wide-Non-Waterproof/dp/B099N8WPRL)|
|[IR_Light_Cage_Top.stl](https://github.com/jones-lab-tamu/CBASv2/blob/main/parts/IR_Light_Cage_Top.stl)| Attach IR to cage | 1 per cage | PLA, 0.2mm layer height, 15% infill|

<p align="left">
    <img src=".//assets/ir_lid.png" alt="Camera diagram" style="width: 300px; height: auto;">
</p>
<p align="left"> 

## IR illumination assembly
1. Cut IR light stripping into 6" pieces.
2. Put strips on both sides of cage top.
3. Solder hookup wire between strips in series.
4. Solder strips to power jack adapter.
5. Connect power jack adapter to output power splitter.
6. We controlled the IR lights with a 5V TTL signal from Clocklab, but any 5V signal should work.

## Video recording
| Part | Purpose | Needed | Notes |
|:---| :--- |:---|:--- |
|PoE IP camera, I706-POE-Black|Camera|1 per cage|[Amazon Link](https://www.amazon.com/Revotech-Camera-Indoor-Security-I706-POE/dp/B096VK398F) |
|6mm lens |Lens | 1 per cage | [Amazon Link](https://www.amazon.com/Xenocam-Focus-Length-Fixed-Camera/dp/B019S2X4JO/r) |

<p align="left">
    <img src=".//assets/camera_diagram.png" alt="Camera diagram" style="width: 300px; height: auto;">
</p>
<p align="left"> 

## Data transfer
| Part | Purpose | Needed | Notes |
|:---| :--- |:---|:--- |
|PoE network switch, Aruba JL684A#ABB|Communicates data|1 per 24 cages|[Amazon Link](https://www.amazon.com/Aruba-Instant-Ethernet-Switch-JL684A/dp/B08B51GFDR)|
|10Gb SFP to RJ45 transceiver, 10GBase-T|Faster data transfer|2 per switch, 1 per NAS| [Amazon Link](https://www.amazon.com/10Gtek-SFP-10G-T-S-Compatible-10GBase-T-Transceiver/dp/B01KFBFL16?th=1)|
|Cat5 cable|Connects cameras to switch|1 per cage| [Amazon Link](https://www.amazon.com/dp/B00BEC0CZG/) 
|Cat8 cable| Connects switch to PC and NAS| 2 per switch, 1 per NAS| [Amazon Link](https://www.amazon.com/dp/B07QLCHZW1)|

<p align="left">
    <img src=".//assets/connection.png" alt="Camera diagram" style="width: 200px; height: auto;">
</p>
<p align="left"> 

## Machine learning computer
| Part | Purpose | Needed | Notes |
|:---| :--- |:---|:--- |
|Computer monitor||1 per switch||
|Keyboard/mouse||1 per switch||
|Machine learning computer||1 per switch|Recommended specs: AMD Ryzen 9 7900X, 32GB+ DDR5 RAM, 1TB+ SSD, NVIDIA GeForce RTX 4090 24GB|



