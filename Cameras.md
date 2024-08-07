# Setting up RTSP IP cameras

## Required parts (see [here](Recording_setup.md) for details)

- PoE IP cameras
- Uplink adapter (RJ-45 transceiver)
- Network switch
- Cat5 cable
- Cat8 cable
- Computer

## Required software
- [Advanced IP Scanner](https://advanced-ip-scanner.com/download/)
- [VLC](https://videolan.org/vlc)


## 1. Connect the computer to the switch.
<p align="left">
    <img src=".//assets/switch.png" alt="CBAS Diagram" style="width: 300px; height: auto;">
</p>
<p align="left"> 

1. Insert an uplink adapter into one of the uplink adapter ports on the right hand side of the network switch (e.g. top left)
 2. Use a Cat8 ethernet cable to connect the ethernet port on the computer to the uplink adapter on the network switch.

## 2. Assign the computer a static IP address.
 1. In Windows Search, type 'Ethernet settings'
 2. Click Edit to the right of IP assignment
 3. Change the dropdown from Automatic (DHCP) to Manual
 4. Change the IPv4 button from off to on
 5. Type `192.168.1.30` in the IP address field
 6. Type `255.255.255.0` into the Subnet mask field
 7. Leave Gateway blank
 8. Leave Preferred DNS blank
 9. Leave everything else off


## 3. Set up the switch.
 1.  Open a web browser (Chrome, Edge, etc.)
 2.  Type `192.168.1.1` into the web browser and press enter
 3.  The switch should pop up with a log-in screen.
 4.  Click "Sign in or Create an Account" (do not connect the switch to the internet)
 5.  Click Connect
 6.  Sign in with the default username/password (admin, *blank*)
 7.  Enter your new username and password
 8.  Login to the switch using your new username and password
 9.  On the Aruba page, click Setup Network
     1.  Go to the window on the left with IPv4 setup
     2.  Change "Management Address Type" to Static
     3.  Make sure "IP Address" is `192.168.1.1/24`
     4.  Make sure "Subnet" is `255.255.255.0`
 10. Back on the Aruba page, click Routing
     1.  Click ARP Table
     2.  Connect a camera, press Refresh
     3.  Write down the camera's IP address (`192.168.1.XX`, `192.168.1.30` is the computer's IP address)
     4.  Do this for all the cameras

## 4. Verify everything
1. Verify IPs
   1. Open Advanced IP Scanner
   2. Clear the text box (defaults with something like `10.244.01` ...)
   3. Type `192.168.1.1-254` into the text box
   4. Press Scan
      1. It should show several IPs
         1. The PC (`192.168.1.30`)
         2. The switch (`192.168.1.1`)
         3. Each camera that's connected (manufacturer: Jinan Jovision Science & Technology Co., Ltd.; `192.168.1.XX`)
2. Verify cameras work
   1. Open VLC
   2. Select Media
   3. Select Open Network Stream
   4. Type `rtsp://192.168.1.XX:8554/profile0`, where `XX` is the IP of the camera
   5.  Press play
       1.  You should see video!

## 5. Change camera settings
1. Open a web browser (Chrome, Edge, etc.)
2. Type in the IP address of the camera into the web browser (`192.168.1.XX`) and press enter
3. Login to the camera with the default username/password (admin, *blank*)
4. Will pop up with "default password, change?," select OK
5. Enter your new username and password
6. Login to the camera using your new username and password
7. In the camera, click "Video and Audio" on the left tab
8. Underneath Video and Audio click the "Video Stream" tab
   1. Change the Main Stream settings to:
      1. Codec: H265
      2. FPS: 10 (for our recordings, set this up however you like)
      3. Quality: Best
      4. Bitrate Control: VBR
      5. Resolution: 2304 x 1296
      6. Bitrate: 3072
   2. Change the Sub Stream settings to:
      1. Codec: H265
      2. FPS: 10 (for our recordings, set this up however you like)
      3. Quality: Good
      4. Bitrate Control: VBR
      5. Resolution: 720 x 480
      6. Bitrate: 256
      7. Whenever the main stream drops it switches to these settings
9. Click Save
10. Underneath Video and Audio click the "Audio Stream" tab
    1.  **Uncheck** "Enable audio stream"
11. Click Save
12. On the left tab, click Display
13. Change the sliders on the Image tab to the following:
    1.  Brightness: 100
    2.  Contrast: 100
    3.  Saturation: 0 (for black and white; set this up however you like)
    4.  Sharpness: 128
    5.  Mirror, Flip, SmartIR: all **unchecked**
    6.  Rotate: None
    7.  Image Style: Standard
14. Click Save
15. Change the sliders on the Exposure tab to the following:
    1.  Anti-Flicker: Off
    2.  Max exposure time: 1/3
    3.  Min exposure time: 1/100000
16. Click Save
17. Chnage the sliders on the Day&Night tab to the following:
    1.  Switch mode: Auto
    2.  Sensitivity: 4
18. Click Save
19. On the left tab, Click Display, then underneath Display click "OSD" (on screen display)
20. Change the OSD tab to the following
    1.  **Uncheck** large font
    2.  Edit name position, time position, and time format to your preference -- this will be the video "overlay" that will not actually be recorded.
21. On the left tab, click Privacy Mask and make sure "Enable privacy mask" is **unchecked**
22. On the left tab, click Alarm, then underneath Alarm click "Motion Detection" and make sure "Enable motion" is **unchecked**


## 6. Finalizing cameras
1. Once you have all your cameras connected to the switch, verify their IP addresses using the Advanced IP Camera program.
2. Log into each camera individually
3. On the left tab, click Network, then underneath network click Basic
4. Change the TCP/IP tab on the right to the following:
   1. DHCP: **unchecked** (this will ungray the text boxes below)
   2. CloudSEE1.0 Compatibility Mode: checked
   3. Auto online/offline: **unchecked**
   4. IP self-adaption: **unchecked**
   5. Manually change the IP address of the camera to an IP address outside the camera range in the Advanced IP scanner program
      1. For instance, if your 24 cameras are 192.168.1.26 to 50, edit each camera so they are 192.168.1.51 to 192.168.1.75.
   6. Click Lock IP
   7. Click Save

## 7. Adding cameras to CBAS

<p align="center">
    <img src=".//assets/acquisition_1.png" alt="CBAS Diagram" style="width: 500px; height: auto;">
</p>
<p align="center"> 

1. Open CBAS
2. Click "Create a Project"
3. Select a main directory where you want your project to exist (e.g. `C:\Users\Jones-Lab\Documents\`)
4. Name your project (e.g. `test_project`) and press Create
5. On the "Record" tab of the CBAS GUI, press the blue circle with the +
6. "Add camera" window will open:
   1. Name the camera (e.g. `cam1`)
   2. Enter the camera's location like this: `rtsp://username:password@192.168.1.XX:8554/profile0`
      1. Example: `rtsp://admin:testpwd@192.168.1.51:8554/profile0`
    3. Press "Add"
7. The camera should appear in the GUI with a screenshot of what the camera's recording
8. Do this for each camera
9. Click the crop button on the camera's screenshot on the GUI
10. Change settings as needed:
    1. Framerate: Should default to whatever you set up in Step 5 above, shouldn't need to change.
    2. Resolution: Defaults to 256, don't change
    3. Crop X, Crop Y, Crop Width, Crop Height: click around these until the camera screenshot is cropped to show the area of the image you care about, the screenshot will stretch and move accordingly
    4. Click Save
11. Click the camera button on each camera's screenshot on the GUI
12. The selected cameras will automatically begin recording
13. Movies will be automatically added to the `\recordings\` folder in the project directory (e.g. `C:\Users\Jones-Lab\Documents\test_project\recordings\`)
    1. Movies will be sorted into folders by date within the recording folder (e.g. `\20240806\`)
    2. Camera recordings will be stored within that folder as .mp4 files (e.g. `\cam1-124553-PM\`)
