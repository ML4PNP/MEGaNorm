IMAGE_NAME = meganorm
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
