input_name=task03

output_name=Кино
output_file=$(output_name).mp4
movie_folder=Gallery/

current_dir=$(dir $(abspath $(firstword $(MAKEFILE_LIST))))
this_folder=$(lastword $(subst /, ,$(current_dir)))
project_dir=$(subst $(this_folder)/,,$(current_dir))
output_dir=$(project_dir)$(movie_folder)
output=$(current_dir)media/videos/task03/1080p60/$(output_file)
movie_dir=$(project_dir)$(movie_folder)

garbage=$(current_dir)media/ $(current_dir)__pycache__/

all: $(output_dir) $(input_name) clean final_message

$(input_name): $(input_name).py
	manim $< --output_file $(output_name)
	@mv $(output) $(movie_dir)


$(output_dir):
	mkdir $@

clean:
	@rm -r $(garbage)

final_message:
	$(info Ваше кино записано в $(movie_dir))
