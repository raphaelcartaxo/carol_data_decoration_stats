

docker_image:
	@docker build -t data-decoration-stats .

docker_run:
	@docker run --rm -it -p 5000:5000 --network="host" --env-file .env data-decoration-stats
