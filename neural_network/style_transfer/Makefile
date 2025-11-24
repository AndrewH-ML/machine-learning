IMAGE=style-transfer-tf:25.02

build:
	docker build -t $(IMAGE) .

run:
	# Run as YOUR host user (avoids root-owned files on the mount)
	docker run --gpus all --rm -it \
	  --user $$(id -u):$$(id -g) \
	  -e HOME=/workspace \
	  -e TF_CPP_MIN_LOG_LEVEL=3 \
	  --shm-size=1g --ulimit memlock=-1 \
	  -v $$(pwd):/workspace \
	  $(IMAGE)

bash:
	docker run --gpus all --rm -it \
	  --user $$(id -u):$$(id -g) \
	  -e HOME=/workspace \
	  -e TF_CPP_MIN_LOG_LEVEL=3 \
	  --shm-size=1g --ulimit memlock=-1 \
	  -v $$(pwd):/workspace \
	  $(IMAGE) bash
