DOCKER_USER ?= smkia
IMAGE_NAME = meganorm
TAG = latest

CONTAINER_NAME = meganorm-container
HOST_PORT = 8888
NOTEBOOK_DIR = $(PWD)/notebooks
RESULTS_DIR = $(PWD)/results
DATA_DIR = $(PWD)/data

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		-p $(HOST_PORT):8888 \
		-v $(NOTEBOOK_DIR):/app/notebooks \
		-v $(RESULTS_DIR):/app/results \
		-v $(DATA_DIR):/app/data \
		$(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME) || true

push:
	@read -p "Enter Docker tag (e.g. latest, v0.1.0): " TAG; \
	docker tag $(IMAGE_NAME) $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)
	docker push $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)


pull:
	docker pull $(DOCKER_USER)/$(IMAGE_NAME):latest
	