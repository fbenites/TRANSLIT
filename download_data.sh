
wget -O data.tar.gz.parta "https://drive.switch.ch/index.php/s/pDYwDIVOg9V9RVW/download"
wget -O data.tar.gz.partb "https://drive.switch.ch/index.php/s/pfLZQeYusy1hV8u/download"
wget -O data.tar.gz.partc "https://drive.switch.ch/index.php/s/6pcMkIrl8NhfZ1U/download"
wget -O data.tar.gz.partd "https://drive.switch.ch/index.php/s/bpXlJK4Ahr3v5xc/download"
wget -O data.tar.gz.parte "https://drive.switch.ch/index.php/s/HdvhVipqaxv9G5t/download"
wget -O data.tar.gz.partf "https://drive.switch.ch/index.php/s/kTCwOmReFytkbZo/download"
wget -O data.tar.gz.partg "https://drive.switch.ch/index.php/s/JK87H254M6yBmwt/download"
wget -O data.tar.gz.parth "https://drive.switch.ch/index.php/s/YaW0COfShNqxiyD/download"
wget -O data.tar.gz.parti "https://drive.switch.ch/index.php/s/XQcZUzatcHvNCn1/download"

cat data.tar.gz.part* > data.tar.gz

rm data.tar.gz.part*
