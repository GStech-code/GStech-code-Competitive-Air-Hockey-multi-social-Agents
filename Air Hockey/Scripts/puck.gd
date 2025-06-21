class_name Puck

extends RigidBody2D

@onready var hit_player_sound: AudioStreamPlayer2D = $HitPlayerSound
@onready var hit_wall_sound: AudioStreamPlayer2D = $HitWallSound


var max_velocity := 3000.0

func _ready():
	body_shape_entered.connect(_on_hit)



func _on_hit(_body_rid: RID, body: Node, _body_shape_index: int, _local_shape_index: int) -> void:
	if body is not Player:
		hit_wall_sound.play()
	elif body is Player:
		hit_player_sound.play()

func _integrate_forces(state: PhysicsDirectBodyState2D) -> void:
	state.linear_velocity = state.linear_velocity.limit_length(max_velocity)
