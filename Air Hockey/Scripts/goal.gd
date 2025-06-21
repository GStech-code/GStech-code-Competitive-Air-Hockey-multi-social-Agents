# Goal.gd
class_name Goal
extends Area2D

signal goal()


func _ready():
	print("✅ Goal script ready on:", name)
	connect("body_entered", Callable(self, "_on_body_entered"))

func _on_body_entered(body: Node):
	print("🚨 Body entered", name, ":", body)
	if body is Puck:
		goal.emit()
		body.queue_free()
