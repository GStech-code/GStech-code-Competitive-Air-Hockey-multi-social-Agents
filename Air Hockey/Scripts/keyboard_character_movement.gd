extends CharacterBody2D


var push_force := 600.0 # Determines how strong is the force of the collision
const SPEED = 3000.0
const ACCEL = 2.0
@onready var sprite: Sprite2D = $Sprite2D
enum P_Color {BLUE, RED}
@export var Paddle_Color: P_Color

func _physics_process(_delta: float) -> void:
	var input_vector := Vector2.ZERO

	# Handle input
	if Paddle_Color == P_Color.BLUE:
		input_vector.x = Input.get_action_strength("P1_Move_Right") - Input.get_action_strength("P1_Move_Left")
		input_vector.y = Input.get_action_strength("P1_Move_Down") - Input.get_action_strength("P1_Move_Up")
	else:
		input_vector.x = Input.get_action_strength("P2_Move_Right") - Input.get_action_strength("P2_Move_Left")
		input_vector.y = Input.get_action_strength("P2_Move_Down") - Input.get_action_strength("P2_Move_Up")

	# Normalize input to avoid faster diagonal movement
	input_vector = input_vector.normalized()

	# Set velocity
	velocity = lerp(velocity, input_vector * SPEED, _delta * ACCEL)

	# Move and detect collisions
	move_and_slide()

	# Push RigidBody2D objects on collision
	for i in range(get_slide_collision_count()):
		var collision := get_slide_collision(i)
		if collision.get_collider() is RigidBody2D:
			collision.get_collider().apply_central_impulse(-collision.get_normal() * push_force)
