all: add_data merge_data selection

add_data: Spotify_Youtube.csv
	./run_add_to_data.sh
	@echo "Add data completed."

merge_data:
	@echo "Merge data completed."

data/data.csv:
	python script/merge_data.py
	@echo "Data merging completed."

selection: data/data_selected.csv
	@echo "Selection completed."

data/data_selected.csv: data/data.csv
	python script/selection.py
	@echo "Data selection completed."

clean:
	rm -f ../data/data_selected.csv
	@echo "Cleanup completed."
