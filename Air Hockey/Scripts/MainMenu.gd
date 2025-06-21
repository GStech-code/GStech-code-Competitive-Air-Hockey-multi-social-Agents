# MainMenue.gd

extends Control

signal start_game_requested(num_games: int)

@onready var num_games_selector: OptionButton = $NumGamesSelector
@onready var start_button: Button = $StartButton

func _ready():
	num_games_selector.clear()
	for i in range(1, 10):
		num_games_selector.add_item(str(i) + " games", i)
	start_button.pressed.connect(_on_start_button_pressed)

func _on_start_button_pressed():
	var selected_num_games := num_games_selector.get_selected_id()
	print("📨 Emitting signal to start game with", selected_num_games, "rounds")
	start_game_requested.emit(selected_num_games)
