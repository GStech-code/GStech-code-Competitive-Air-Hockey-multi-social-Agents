# MainMenue.gd

extends Control

@onready var num_games_selector: OptionButton = $NumGamesSelector
@onready var start_button: Button = $StartButton

func _ready():
	num_games_selector.clear()
	for i in range(1, 10):
		num_games_selector.add_item(str(i) + " games", i)

	start_button.pressed.connect(_on_start_button_pressed)

func _on_start_button_pressed():
	var selected_num_games := num_games_selector.get_selected_id()
	var game_scene := preload("res://scenes/Game.tscn").instantiate()
	game_scene.num_games = selected_num_games

	get_parent().add_child(game_scene)  # cleaner than get_tree().get_root()
	queue_free()  # remove the menu once the game starts
