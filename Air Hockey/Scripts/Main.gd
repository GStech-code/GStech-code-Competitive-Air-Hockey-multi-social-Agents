# Main.gd

extends Node

@onready var music_bus_index := AudioServer.get_bus_index("MusicBus")
var current_scene: Node = null  # track current child scene
@onready var music_player: AudioStreamPlayer = $MusicPlayer

func _ready():
	_load_main_menu()

func _load_main_menu():
	var menu = preload("res://Scenes/MainMenu.tscn").instantiate()
	menu.start_game_requested.connect(_on_start_game_requested)
	_replace_scene(menu)
	_set_music_filter(true)

func _on_start_game_requested(num_games: int):
	print("🎮 Starting game with", num_games, "rounds")
	var game = preload("res://Scenes/Game.tscn").instantiate()
	game.num_games = num_games
	game.game_paused.connect(_game_paused)
	game.rerurn_to_menue_requested.connect(_on_rerurn_to_menue_requested)
	game.start_game_requested.connect(_on_start_game_requested)
	_replace_scene(game)
	_set_music_filter(false)

func _game_paused():
	_set_music_filter(true)

func _on_rerurn_to_menue_requested():
	print("🎮Returning to the main menue")
	get_tree().paused = false
	_set_music_filter(true)
	_load_main_menu()


func _replace_scene(scene: Node):
	if current_scene:
		current_scene.queue_free()
	current_scene = scene
	add_child(scene)

func _set_music_filter(enable: bool):
	if AudioServer.get_bus_effect_count(music_bus_index) > 0:
		AudioServer.set_bus_effect_enabled(music_bus_index, 0, enable)
