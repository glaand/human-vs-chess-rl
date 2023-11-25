run:
	@export PYTHONPATH=$PYTHONPATH:$(pwd)
	@python omegazero.py

clean_games:
	@rm -rf games/*

clean_artifacts:
	@rm -rf artifacts/*

clean_log:
	@rm -rf log.txt

clean_plots:
	@rm -rf *.png

clean_csv:
	@rm -rf *.csv

clean: clean_games clean_artifacts clean_plots clean_log clean_csv