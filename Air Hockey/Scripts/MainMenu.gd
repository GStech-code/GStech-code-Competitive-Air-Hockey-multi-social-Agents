# MainMenue.gd

extends Control

# Custom signal to notify when the game should start, passing number of rounds
signal start_game_requested(num_games: int)

# UI Node References
@onready var num_games_selector: OptionButton = $NumGamesSelector
@onready var start_button: Button = $StartButton
# Sound Nodes
@onready var num_games_hover_sound: AudioStreamPlayer = $NumGamesSelector/NumGamesHoverSound
@onready var num_games_selected_sound: AudioStreamPlayer = $NumGamesSelector/NumGamesSelectedSound
@onready var start_button_sound: AudioStreamPlayer = $StartButton/StartButtonSound

#  Access the internal PopupMenu to handle dropdown logic
@onready var popup_menu: PopupMenu = num_games_selector.get_popup()
var last_hovered_item := -1

func _ready():
	# Clear previous items (if any)
	num_games_selector.clear()
	# Populate the OptionButton with 1–9 game choices
	for i in range(1, 10):
		num_games_selector.add_item(str(i) + " games", i)
		# Populate the OptionButton with 1–9 game choices
	start_button.pressed.connect(_on_start_button_pressed)
	# Connect to item selection for playing the "selected" sound
	num_games_selector.mouse_entered.connect(_on_num_games_selector_hovered)
	# Connect to the internal PopupMenu's signal to know when the dropdown opens
	num_games_selector.item_selected.connect(_on_num_games_selector_pressed)


func _on_start_button_pressed():
	# Play sound when the start button is pressed
	start_button_sound.play()
	# Delay before starting the game (for UX)
	await get_tree().create_timer(0.5).timeout
	# Get the selected number of games and emit the signal
	var selected_num_games := num_games_selector.get_selected_id()
	print("📨 Emitting signal to start game with", selected_num_games, "rounds")
	start_game_requested.emit(selected_num_games)


func _on_num_games_selector_pressed(_index):
	# Play the selection sound when an item is chosen
	num_games_selected_sound.play()


func _on_dropdown_opened():
	# This is called when the OptionButton's dropdown opens.
	# We hook into the global unhandled input system for tracking mouse movement over items.
	# We use CONNECT_ONE_SHOT so this only happens while the dropdown is open.
	get_tree().connect("unhandled_input", _on_unhandled_input_for_hover, CONNECT_ONE_SHOT)


func _on_unhandled_input_for_hover(event: InputEvent):
	# Only process mouse motion events while the dropdown is open
	if event is InputEventMouseMotion and popup_menu.visible:
		# Get the mouse position relative to the PopupMenu
		var mouse_pos = popup_menu.get_local_mouse_position()
		# Get the index of the item currently under the mouse
		var item_index = popup_menu.get_item_at_position(mouse_pos)
		
		if item_index != -1 and item_index != last_hovered_item:
			# Item found under cursor — play hover sound
			# you may want to track the last hovered index to avoid spamming sound
			last_hovered_item = item_index
			num_games_hover_sound.play()


func _on_num_games_selector_hovered():
		num_games_hover_sound.play()
