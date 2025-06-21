# Game.gd

extends Node2D

signal start_game_requested(num_games: int)
signal rerurn_to_menue_requested()
signal game_paused(is_paused: bool)

var scores = {"left": 0, "right": 0}

@export var num_games: int = 3
@export var puck_scene: PackedScene
@export var puck_respawn_position: Vector2 = Vector2(3296, 1854)

@onready var score_left_label: Label = %Score_Left_label
@onready var score_right_label: Label = %Score_Right_label

@onready var puck: Puck = $puck

@onready var pause_ui: CanvasLayer = $PauseUI
@onready var control: Control = $PauseUI/Control
@onready var winner_label: Label = $PauseUI/Control/Panel/WinnerLabel
@onready var restart_button: Button = $"PauseUI/Control/Panel/Restart Button"
@onready var main_menue_button: Button = $"PauseUI/Control/Panel/MainMenue Button"



var game_over := false

func _set_ui_pause_mode():
	var nodes := [
		pause_ui,
		control,
		$PauseUI/Control/Panel,
		winner_label,
		restart_button
	]
	for node in nodes:
		if "pause_mode" in node:
			node.pause_mode = 2
			print("Set pause mode for", node.name, "→", node.pause_mode)
		else:
			print("⚠️ Skipped:", node.name, "(no pause_mode)")


func _ready():
	await get_tree().process_frame  # Wait one frame to ensure everything's in the scene
	puck_scene = preload("res://scenes/Puck.tscn")
	# Ensure UI nodes still work when game is paused
	_set_ui_pause_mode()
	# Connect restart button
	game_over = false
	scores["right"] = 0
	scores["left"] = 0
	pause_ui.visible = false
	_respawn_puck()
	restart_button.pressed.connect(_on_restart_button_pressed)
	

func _on_restart_button_pressed():
	get_tree().paused = false
	start_game_requested.emit(num_games)
	
func _on_main_menue_button_pressed():
	print("📨 Emitting signal to return to the main menue")
	rerurn_to_menue_requested.emit()

	
func _end_game(winning_side: String):
	main_menue_button.pressed.connect(_on_main_menue_button_pressed)
	game_over = true
	if puck:
		puck.queue_free()

	winner_label.text = winning_side + " Wins!"
	pause_ui.visible = true
	get_tree().paused = true
	game_paused.emit()
	
	
	


func _on_left_goal_goal() -> void:
	if game_over:
		print("Game is over, not respawning puck.")
		return

	scores["right"] += 1
	print("Right scored! New score:", scores["right"])
	score_right_label.text = str(scores["right"])

	if scores["right"] >= num_games:
		print("Right wins the game!")
		_end_game("Right")
	else:
		if puck:
			print("Queue freeing puck...")
			puck.queue_free()
			puck = null
		await get_tree().create_timer(1.0).timeout
		print("Calling _respawn_puck()")
		_respawn_puck()


func _on_right_goal_goal() -> void:
	if game_over:
		print("Game is over, not respawning puck.")
		return

	scores["left"] += 1
	print("Left scored! New score:", scores["left"])
	score_left_label.text = str(scores["left"])

	if scores["left"] >= num_games:
		print("Left wins the game!")
		_end_game("Left")
	else:
		if puck:
			print("Queue freeing puck...")
			puck.queue_free()
			puck = null
		await get_tree().create_timer(1.0).timeout
		print("Calling _respawn_puck()")
		_respawn_puck()


func _respawn_puck():
	if puck_scene:
		await get_tree().process_frame  # give queue_free() time to take effect
		
		# sanity check
		if puck:
			print("⚠️ Warning: puck still exists after queue_free(), freeing again")
			puck.queue_free()
			puck = null
			await get_tree().process_frame

		print("⚽ Respawning puck...")
		var new_puck = puck_scene.instantiate() as Puck
		new_puck.global_position = puck_respawn_position
		new_puck.linear_velocity = Vector2.ZERO
		new_puck.angular_velocity = 0.0
		add_child(new_puck)
		puck = new_puck
		print("✅ Puck respawned at:", puck.global_position)
		print("Children of Game after puck spawn:")
		for child in get_children():
			print("- ", child.name)
