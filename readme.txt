# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/ChaoningZhang/MobileSAM

# How to run
python app.py


# Set server:
connect rpi and server in the same network first.
then find the rpi IP address.
RASPI_URL = os.environ.get("RASPI_URL", "http://192.168.1.100:8000")  # change to your Pi IP