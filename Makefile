DOCKER_USER = smkia
IMAGE_NAME = meganorm
TAG = latest
IMAGE = $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)

CONTAINER_NAME = meganorm-container
HOST_PORT = 8888
NOTEBOOK_DIR = $(PWD)/notebooks
RESULTS_DIR = $(PWD)/results
DATA_DIR = $(PWD)/data

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		-p 127.0.0.1:$(HOST_PORT):8888 \
		-v $(NOTEBOOK_DIR):/app/notebooks \
		-v $(RESULTS_DIR):/app/results \
		-v $(DATA_DIR):/app/data \
		$(IMAGE)

stop:
	docker stop $(CONTAINER_NAME) || true

push:
	docker push $(IMAGE)


pull:
	docker pull $(IMAGE)

.PHONY: build run stop push pull
