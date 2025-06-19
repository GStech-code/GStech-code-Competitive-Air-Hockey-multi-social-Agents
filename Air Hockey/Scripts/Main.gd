# Main.gd
extends Node

func _ready():
	var main_menu_scene := preload("res://Scenes/MainMenu.tscn").instantiate()
	add_child(main_menu_scene)
