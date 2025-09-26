
#!/bin/bash
#YOLO runner with loggin

LOGFILE=/home/admin/yolo.log

echo "=== Starting YOLO at $(date) ===" >> $LOGFILE

# Change into YOLO directory
YOLO_DIR=/home/admin/yolo
if [ -d "$YOLO_DIR" ]; then
cd "$YOLO_DIR"
else
echo "Error: YOLO directory $YOLO_DIR not found!" >> $LOGFILE
exit 1
fi

# Activate virtual environment 
VENV_DIR="$YOLO_DIR/venv"
if [ -d "$VENV_DIR" ]; then
source "$VENV_DIR/bin/activate" >> $LOGFILE 2>&1
else
echo "Error: virtual environment $VENV_DIR not found!" >> $LOGFILE
exit 1
fi

#Run YOLO script with live feed
python grater.py \
--model=yolo11n_ncnn_model \
--source=usb0 \
--resolution=1200x720 >> $LOGFILE 2>&1

#Timestamp at end
echo "=== YOLO stopped at $(date) ===" >> $LOGFILE
