
wget -O data.tar.gz.parta "https://filer.cloudlab.zhaw.ch/index.php/s/SaJRpdBnxArbP2f/download"
wget -O data.tar.gz.partb "https://filer.cloudlab.zhaw.ch/index.php/s/t2FMbgHxfDyXXpP/download"
wget -O data.tar.gz.partc "https://filer.cloudlab.zhaw.ch/index.php/s/4fmf9ffstM6kTJo/download"
wget -O data.tar.gz.partd "https://filer.cloudlab.zhaw.ch/index.php/s/EdkWSnCcCEiS9W6/download"
wget -O data.tar.gz.parte "https://filer.cloudlab.zhaw.ch/index.php/s/ZdF4ayqjLsF63wr/download"
wget -O data.tar.gz.partf "https://filer.cloudlab.zhaw.ch/index.php/s/ec4wz9GpewACXYn/download"
wget -O data.tar.gz.partg "https://filer.cloudlab.zhaw.ch/index.php/s/zEaAjpwbM88XGYa/download"
wget -O data.tar.gz.parth "https://filer.cloudlab.zhaw.ch/index.php/s/ems76MdDGgBAxzr/download"
wget -O data.tar.gz.parti "https://filer.cloudlab.zhaw.ch/index.php/s/XQAKrL6pwSWrRAP/download"

cat data.tar.gz.part* > data.tar.gz

rm data.tar.gz.part*
